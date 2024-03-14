import os
import math
from tqdm import trange, tqdm
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from typing import Union, Callable, Dict, Optional

from src.models.model_heads import ClfHead
from src.models.model_base import BaseModel
from src.models.model_bert import BertModel
from src.training_logger import TrainLogger
from src.utils import dict_to_device, evaluate_model, mixstyle
import numpy as np


class TaskModel(BertModel):

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = .3,
        n_hidden: int = 0,
        bottleneck: bool = False,
        bottleneck_dim: Optional[int] = None,
        bottleneck_dropout: Optional[float] = None,
        wandb_logger=None,
        augment=False,
        no_scheduler=False,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)

        self.num_labels = num_labels
        self.dropout = dropout
        self.n_hidden = n_hidden
        self.has_bottleneck = bottleneck
        self.bottleneck_dim = bottleneck_dim
        self.bottleneck_dropout = bottleneck_dropout
        self.wandb_logger = wandb_logger
        self.global_step = 0
        self.augment = augment
        self.no_scheduler = no_scheduler
        # bottleneck layer
        # if self.has_bottleneck:
        #     self.bottleneck = ClfHead(self.hidden_size, bottleneck_dim, dropout=bottleneck_dropout)
        #     self.in_size_heads = bottleneck_dim
        # else:
        self.bottleneck = torch.nn.Identity()
        self.in_size_heads = self.hidden_size

        self.task_head = ClfHead([self.in_size_heads]*(n_hidden+1), num_labels, dropout=dropout)

    def param_spec(self):
        num_param = 0
        trainable = 0
        frozen = 0
        for param in self.parameters():
            if param.requires_grad == True:
                trainable += len(param.reshape(-1))
            else:
                frozen += len(param.reshape(-1))
            num_param += len(param.reshape(-1))
        percentage = np.round(trainable / num_param * 100, 2)
        log = {"total_param": num_param, "trainable param": trainable,
               "frozen param": frozen, "trainable pecentage": percentage}
        return log

    def _forward(self,
                 y=None,
                 past_key_values=None,
                 head_mask=None,
                 **x) -> torch.Tensor:
        input_shape = x["input_ids"].size()
        batch_size, seq_length = input_shape
        token_type_ids = torch.zeros_like(x["input_ids"]) if "token_type_ids" not in x.keys() else x["token_type_ids"]
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(x["attention_mask"], input_shape)
        head_mask = self.bert.get_head_mask(head_mask, self.bert.config.num_hidden_layers)
        if self.augment and (y is not None):
            embed = self.bert.embeddings(input_ids=x["input_ids"],
                                         token_type_ids=token_type_ids,
                                         past_key_values_length=past_key_values_length)
            old_embed = embed.detach()
            for i in torch.unique(y):
                # print(i, y)
                mask = y == i
                # print(mask)
                x_ = embed[mask]
                # old_embed = embed[mask].detach()
                embed[mask] = mixstyle(x_)
                # new_embed = embed[mask].detach()
                # print(torch.equal(old_embed, new_embed))
            # new_embed = embed.detach()
            # x = {"hidden_states": embed, "encoder_attention_mask": x["attention_mask"]}
            # for i in range(len(self.bert.encoder.layer)):
            #     x["hidden_states"] = self.bert.encoder.layer[i](**x)[0]
            # hidden = x["hidden_states"][:, 0]
            x = {"hidden_states": embed, "attention_mask": extended_attention_mask}
            for i in range(len(self.bert.encoder.layer)):
                layer_head_mask = head_mask[i] if head_mask is not None else None
                past_key_value = past_key_values[i] if past_key_values is not None else None
                x["head_mask"] = layer_head_mask
                x["past_key_value"] = past_key_value
                x["hidden_states"] = self.bert.encoder.layer[i](**x)[0]
            hidden = x["hidden_states"][:, 0]
        else:
            hidden = self.bert(**x).last_hidden_state[:, 0]

        return self.bottleneck(hidden)
    # def _forward(self, **x) -> torch.Tensor:
    #     hidden = super()._forward(**x)
    #     return self.bottleneck(hidden)

    def forward(self, y=None, **x) -> torch.Tensor:
        return self.task_head(self._forward(y=y, **x))

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: TrainLogger,
        loss_fn: Callable,
        pred_fn: Callable,
        metrics: Dict[str, Callable],
        num_epochs: int,
        learning_rate: float,
        learning_rate_bottleneck: float,
        learning_rate_head: float,
        optimizer_warmup_steps: int,
        max_grad_norm: float,
        output_dir: Union[str, os.PathLike],
        cooldown: int,
        checkpoint_name: Optional[str] = None,
        seed: Optional[int] = None
    ) -> None:

        train_steps = len(train_loader) * num_epochs
        self._init_optimizer_and_schedule(
            train_steps,
            learning_rate,
            learning_rate_head,
            learning_rate_bottleneck,
            optimizer_warmup_steps
        )

        self.zero_grad()

        train_str = "Epoch {}, {}"
        str_suffix = lambda d: ", ".join([f"{k}: {v}" for k,v in d.items()])

        performance_decrease_counter = 0
        train_iterator = trange(num_epochs, desc=train_str.format(0, ""), leave=False, position=0)
        for epoch in train_iterator:

            self._step(
                train_loader,
                loss_fn,
                logger,
                max_grad_norm,
            )

            result = self.evaluate(
                val_loader,
                loss_fn,
                pred_fn,
                metrics
            )

            if self.wandb_logger is not None:
                res = result.copy()
                res["lr"] = self.optimizer.param_groups[0]["lr"]
                for key in result.keys():
                    res[f"task_{key}"] = res.pop(key)
                res["epochs"] = epoch
                self.wandb_logger.log(res)
            logger.validation_loss(epoch, result, "task")

            train_iterator.set_description(
                train_str.format(epoch, str_suffix(result)), refresh=True
            )

            if logger.is_best(result["bacc"], ascending=False):
                cpt = self.save_checkpoint(Path(output_dir), checkpoint_name, seed)
                cpt_result = result
                cpt_epoch = epoch
                performance_decrease_counter = 0
            else:
                performance_decrease_counter += 1

            if performance_decrease_counter>cooldown:
                break

        print("Final result after " + train_str.format(epoch, str_suffix(result)))
        print("Best result: " + train_str.format(cpt_epoch, str_suffix(cpt_result)))

        return cpt


    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        loss_fn: Callable,
        pred_fn: Callable,
        metrics: Dict[str, Callable]
    ) -> dict:
        self.eval()

        forward_fn = lambda x: self(**x)
        result = evaluate_model(
            self,
            val_loader,
            loss_fn,
            pred_fn,
            metrics,
            forward_fn=forward_fn,
        )
        result["g_step"] = self.global_step

        if self.wandb_logger:
            self.wandb_logger.log(result)
        return result

    @torch.no_grad()
    def calculate_gap(self,
                        val_loader: DataLoader,
    ) -> dict:

        self.eval()
        dev = next(self.parameters()).device

        result = {}
        forward_fn = lambda x: self.forward(x)

        GAP_1 = 0.
        gap = []
        val_iterator = tqdm(val_loader, desc=f"evaluating {desc}", leave=False, position=1)
        btch = next(iter(val_loader))
        # if len(btch) > 2


        for i, batch in enumerate(val_iterator):
            inputs, labels = batch[input_idx], batch[label_idx]
            if isinstance(inputs, dict):
                inputs = dict_to_device(inputs, dev)
            else:
                inputs = inputs.to(dev)
            if hasattr(model, "forward_w"):
                logits = forward_fn(w, inputs)
            else:
                logits = forward_fn(inputs)
            if isinstance(logits, list):
                eval_loss += torch.stack([loss_fn(x, labels.to(dev)) for x in logits]).mean().item()
                preds, _ = torch.mode(torch.stack([pred_fn(x.cpu()) for x in logits]), dim=0)
            else:
                eval_loss += loss_fn(logits, labels.to(dev)).item()
                preds = pred_fn(logits.cpu())
            output_list.append((
                preds,
                labels
            ))

        p, l = list(zip(*output_list))
        predictions = torch.cat(p, dim=0)
        labels = torch.cat(l, dim=0)
        if hasattr(model, "forward_w"):
            for metric_name, metric in metrics.items():
                result[f"{metric_name}_{w}"] = metric(predictions, labels)
            result[f"loss_{w}"] = eval_loss / (i + 1)
        else:
            for metric_name, metric in metrics.items():
                result[f"{metric_name}"] = metric(predictions, labels)
            result[f"loss"] = eval_loss / (i + 1)
        return result
        return result

    def _step(
        self,
        train_loader: DataLoader,
        loss_fn: Callable,
        logger: TrainLogger,
        max_grad_norm: float
    ) -> None:
        self.train()

        epoch_str = "training - step {}, loss: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan), leave=False, position=1)

        for step, batch in enumerate(epoch_iterator):

            inputs, labels = batch[0], batch[1]
            inputs = dict_to_device(inputs, self.device)
            if self.augment:
                outputs = self(y=labels, **inputs)
            else:
                outputs = self(**inputs)

            loss = loss_fn(outputs, labels.to(self.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.bert.parameters(), max_grad_norm)

            self.optimizer.step()
            if not self.no_scheduler:
                self.scheduler.step()
            # self.scheduler.step()
            self.zero_grad()
            losses_dict = {
                f"train loss": loss.item(),
                "g_step": self.global_step
            }
            # print(loss.item())
            if self.wandb_logger:
                self.wandb_logger.log(losses_dict)

            logger.step_loss(self.global_step, {"total": loss.item(), "task": loss.item()})

            epoch_iterator.set_description(epoch_str.format(step, loss.item()), refresh=True)

            self.global_step += 1


    def _init_optimizer_and_schedule(
        self,
        num_training_steps: int,
        learning_rate: float,
        learning_rate_head: float,
        learning_rate_bottleneck: float = 1e-4,
        num_warmup_steps: int = 0
    ) -> None:
        optimizer_params = [
            {
                "params": self.bert.parameters(),
                "lr": learning_rate
            },
            {
                "params": self.bottleneck.parameters(),
                "lr": learning_rate_bottleneck
            },
            {
                "params": self.task_head.parameters(),
                "lr": learning_rate_head
            }
        ]

        self.optimizer = AdamW(optimizer_params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)

        if not self.no_scheduler:
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                             num_warmup_steps=num_warmup_steps,
                                                             num_training_steps=num_training_steps )


    def make_checkpoint_name(
        self,
        seed: Optional[int] = None
    ):
        filename_parts = [
            self.model_name.split('/')[-1],
            "task_baseline",
            "cp_init" if self.state_dict_init else None,
            f"seed{seed}" if seed is not None else None
        ]
        return "-".join([x for x in filename_parts if x is not None]) + ".pt"


    def save_checkpoint(
        self,
        output_dir: Union[str, os.PathLike],
        checkpoint_name: Optional[str] = None,
        seed: Optional[int] = None
    ) -> None:
        info_dict = {
            "cls_name": self.__class__.__name__,
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "dropout": self.dropout,
            "n_hidden": self.n_hidden,
            "bottleneck": self.has_bottleneck,
            "bottleneck_dim": self.bottleneck_dim,
            "bottleneck_dropout": self.bottleneck_dropout,
            "encoder_state_dict": self.bert.state_dict(),
            "bottleneck_state_dict": self.bottleneck.state_dict(),
            "task_head_state_dict": self.task_head.state_dict()
        }

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if checkpoint_name is None:
            checkpoint_name = self.make_checkpoint_name(seed)
        filepath = output_dir / checkpoint_name
        torch.save(info_dict, filepath)
        return filepath


    @classmethod
    def load_checkpoint(
        cls,
        filepath: Union[str, os.PathLike],
        map_location: Union[str, torch.device] = torch.device('cpu')
    ) -> torch.nn.Module:
        info_dict = torch.load(filepath, map_location=map_location)

        cls_instance = cls(
            info_dict['model_name'],
            info_dict['num_labels'],
            info_dict['dropout'],
            info_dict['n_hidden'],
            info_dict['bottleneck'],
            info_dict['bottleneck_dim'],
            info_dict['bottleneck_dropout']
        )
        cls_instance.bert.load_state_dict(info_dict['encoder_state_dict'])
        cls_instance.bottleneck.load_state_dict(info_dict['bottleneck_state_dict'])
        cls_instance.task_head.load_state_dict(info_dict['task_head_state_dict'])

        cls_instance.eval()

        return cls_instance