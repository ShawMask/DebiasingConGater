import os
import math
from tqdm import trange, tqdm
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.functional import triplet_margin_loss
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from sklearn.metrics import balanced_accuracy_score
from collections import OrderedDict
import numpy as np

from typing import Union, Callable, Dict, Optional
from itertools import product
from sklearn import svm

from src.models.model_heads import ClfHead, AdvHead
from src.models.model_bert import BertModel
from src.training_logger import TrainLogger
from src.utils import dict_to_device, congater_evaluate_model, get_mean_loss


# Trajectory Sigmoid Activation Function
class Tsigmoid(nn.Module):
    def __init__(self):
        super(Tsigmoid, self).__init__()
    def forward(self, x, w):
        if type(w) is torch.tensor:
            out = 1 - torch.log2(w + 1) / (1 + torch.exp(x))  # 0<w<1
        else:
            out = 1 - torch.log2(torch.tensor(w+1)) / (1 + torch.exp(x))  # 0<w<1
        return out


# ConGater Can be a single layer or multiple layers followed by T-Sgimoid Activation Function
class GateLayer(nn.Module):
    def __init__(self, embed_size, num_layers=2, squeeze_ratio=4, dropout=0.0):
        super(GateLayer, self).__init__()
        self.num_layers = num_layers
        embed_size = [embed_size] + [embed_size//((i+1)*squeeze_ratio) for i in range(num_layers-1)]
        self.linear = nn.ModuleList()
        # Generate linear layer followed by Tanh for hidden layers in case num layer >= 2
        for i in range(num_layers):
            if i < (num_layers - 1):
                self.linear.append(nn.Sequential(nn.Linear(embed_size[i], embed_size[i+1]), nn.Tanh(),
                                                 nn.Dropout(dropout)))
            else:
                self.linear.append(nn.Linear(embed_size[-1], embed_size[0]))

        self.activation = Tsigmoid()


    def forward(self, embed, w):
        for i in range(self.num_layers):
            embed = self.linear[i](embed)
        return self.activation(embed, w)




class ConGaterModel(BertModel):

    def __init__(
        self,
        model_name: str,
        num_labels_task: int,
        num_labels_protected: Union[int, list, tuple],
        task_dropout: float = .3,
        task_n_hidden: int = 0,
        adv_dropout: float = .3,
        adv_n_hidden: int = 1,
        adv_count: int = 5,
        bottleneck: bool = False,
        bottleneck_dim: Optional[int] = None,
        bottleneck_dropout: Optional[float] = None,
        task_head_state_dict: OrderedDict = None,
        task_head_freeze: bool = False,
        encoder_state_dict: OrderedDict = None,
        congater_names: Optional[str] = "gender",
        congater_position: str = "all",
        num_gate_layers: int = 1,
        gate_squeeze_ratio: int = 4,
        default_trainable_parameters: str = "all",
        custom_init: int = 0,
        no_scheduler: bool = False,
        wandb_logger: object = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)

        if isinstance(num_labels_protected, int):
            num_labels_protected = [num_labels_protected]

        self.num_labels_task = num_labels_task
        self.num_labels_protected = num_labels_protected
        self.task_dropout = task_dropout
        self.task_n_hidden = task_n_hidden
        self.adv_dropout = adv_dropout
        self.adv_n_hidden = adv_n_hidden
        self.adv_count = adv_count
        self.has_bottleneck = bottleneck
        self.bottleneck_dim = bottleneck_dim
        self.bottleneck_dropout = bottleneck_dropout
        self.task_head_freeze = task_head_freeze
        self.congater_position = congater_position
        self.num_gate_layers = num_gate_layers
        self.gate_squeeze_ratio = gate_squeeze_ratio
        self.wandb_logger = wandb_logger
        self.global_step = 0
        self.evaluate_w = {"task": 0}
        self.custom_init = custom_init
        self.best_model = None
        self.no_scheduler = no_scheduler
        if congater_names:
            self.congater_names = congater_names if type(congater_names) is list else [congater_names]
            self.evaluate_w = list(product(self.evaluate_w, [0])) + list(product([0], self.evaluate_w))
        else:
            self.congater_names = None

        self.training_stage = ["task"] + self.congater_names if self.congater_names else ["task"]

        self.val_w = [0, 1]
        self.default_trainable_parameters = default_trainable_parameters
        self.num_encoder_layers = len(self.bert.encoder.layer)

        if self.congater_names:
            self.congater = nn.ModuleDict()
            for name in self.congater_names:
                self.congater[name] = nn.ModuleList()
                if self.congater_position == "all":
                    for i in range(self.num_encoder_layers):
                        self.congater[name].append(GateLayer(self.hidden_size, num_layers=num_gate_layers,
                                                             squeeze_ratio=gate_squeeze_ratio,
                                                             dropout=self.adv_dropout))
                        if self.custom_init:
                            with torch.no_grad():
                                if self.num_gate_layers > 1:
                                    for layer in self.congater[name][i].linear:
                                        # print(layer)
                                        if isinstance(layer, nn.Linear):
                                            layer.weight = nn.Parameter(torch.zeros_like(layer.weight),
                                                                        requires_grad=True)
                                            layer.bias = nn.Parameter(torch.ones_like(layer.bias) * 5,
                                                                      requires_grad=True)

                elif self.congater_position == "last":
                    self.congater[name].append(GateLayer(self.hidden_size, num_layers=num_gate_layers,
                                                         squeeze_ratio=gate_squeeze_ratio, ))

        # bottleneck layer
        if self.has_bottleneck:
            self.bottleneck = ClfHead(self.hidden_size, bottleneck_dim, dropout=bottleneck_dropout)
            self.in_size_heads = bottleneck_dim
        else:
            self.bottleneck = torch.nn.Identity()
            self.in_size_heads = self.hidden_size

        # heads
        self.task_head = ClfHead([self.in_size_heads]*(task_n_hidden+1), num_labels_task, dropout=task_dropout)

        if task_head_state_dict is not None:
            self.task_head.load_state_dict(task_head_state_dict)
            if task_head_freeze:
                for p in self.task_head.parameters():
                    p.requires_grad = False

        self.adv_head = torch.nn.ModuleList()
        for n in num_labels_protected:
            self.adv_head.append(
                AdvHead(adv_count, hid_sizes=[self.in_size_heads]*(adv_n_hidden+1), num_labels=n, dropout=adv_dropout)
            )

        self.set_trainable_parameters(self.default_trainable_parameters)

    # The Parameters of the model can be modified with this method and strings
    # Example: set_trainable_parameters("encoder 11+head") sets any parameter in 11th encoder layer and head to True the rest to false
    # + is the sign of or and " " (space) is the sign for "and"
    # it can take up to 3 or (+) signs
    def set_trainable_parameters(self, trainable_param):
        ln_gate = False
        if trainable_param == "none":
            for param in self.parameters():
                param.requires_grad = False

        elif "only_output" in trainable_param:
            trainable_param = "_".join(trainable_param.split("_")[1:])
            for name, param in self.named_parameters():
                # print(trainable_param, name, self.condition_(trainable_param, name))
                if self.condition_(trainable_param, name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            for name, param in self.named_parameters():
                if ('output' in name) and ('attention' in name):
                    param.requires_grad = False

        elif trainable_param == 'all':
            for param in self.parameters():
                param.requires_grad = True
        else:
            for name, param in self.named_parameters():
                # print(trainable_param, name, self.condition_(trainable_param, name))
                if self.condition_(trainable_param, name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
    # how conditions are combined to make the final decision of the paramrter training
    def condition_(self, param, text):
        if '+' in param:
            a = param.split('+')
            b = []
            for t in a:
                b.append(t.split(' '))
        else:
            b = param.split(' ')
        if type(b[0]) == str:
            if all(word in text for word in b):
                return True
            else:
                return False
        if len(b) == 2:
            if all(word in text for word in b[0]) or all(word in text for word in b[1]):
                return True
            else:
                return False
        if len(b) == 3:
            if all(word in text for word in b[0]) or all(word in text for word in b[1]) or all(
                    word in text for word in b[2]):
                return True
            else:
                return False

    def print_trainable_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)
    #logs parameter details
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
        percentage = np.round(trainable / num_param * 100, 1)
        log = {"total_param": num_param, "trainable param": trainable,
               "frozen param": frozen, "trainable pecentage": percentage}
        return log
    # For a given omega calculates the gate value of the embeddings
    def forward_gate(self, x, w, layer):
        x_gate = []
        for name in self.congater_names:
            x_gate.append(self.congater[name][layer](x, w[name]))
        if len(x_gate) > 1:
            x_gate = [x_gate[i] * x_gate[i - 1] for i in range(1, len(x_gate))][0]
        else:
            x_gate = x_gate[0]

        return x_gate

    # Follows through transformer block to calculate the final embeddings based on ConGater Layers
    def _forward(self, w,
                 past_key_values=None,
                 y=None,
                 head_mask=None,
                 interpolate=False,
                 **x) -> torch.Tensor:

        if self.congater_position == "all":
            input_shape = x["input_ids"].size()
            batch_size, seq_length = input_shape
            token_type_ids = torch.zeros_like(x["input_ids"]) if "token_type_ids" not in x.keys() else x["token_type_ids"]
            past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(x["attention_mask"], input_shape)
            head_mask = self.bert.get_head_mask(head_mask, self.bert.config.num_hidden_layers)

            embed = self.bert.embeddings(input_ids=x["input_ids"],
                                         token_type_ids=token_type_ids,
                                         past_key_values_length=past_key_values_length
                                         )

            x = {"hidden_states": embed, "attention_mask": extended_attention_mask}
            for i in range(len(self.bert.encoder.layer)):
                layer_head_mask = head_mask[i] if head_mask is not None else None
                past_key_value = past_key_values[i] if past_key_values is not None else None
                x["head_mask"] = layer_head_mask
                x["past_key_value"] = past_key_value

                x["hidden_states"] = self.bert.encoder.layer[i](**x)[0]*self.forward_gate(x["hidden_states"], w, i)



            hidden = x["hidden_states"][:, 0]

        else:  # self.congater_position == "last":
            hidden = self.bert(**x)
            hidden = hidden * self.forward_gate(hidden, w, -1)

        return self.bottleneck(hidden)

    # Task Logit Calculation
    def forward(self, w, interpolate=False, **x) -> torch.Tensor:
        return self.task_head(self._forward(w, interpolate=interpolate, **x))

    # Attribute Logic Calculation based on Head
    def forward_protected(self, head_idx, w, interpolate=False, **x) -> torch.Tensor:
        return self.adv_head[head_idx](self._forward(w, interpolate=interpolate, **x))


    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: TrainLogger,
        loss_fn: Callable,
        pred_fn: Callable,
        metrics: Dict[str, Callable],
        loss_fn_protected: Union[Callable, list, tuple],
        pred_fn_protected: Union[Callable, list, tuple],
        metrics_protected: Union[Dict[str, Callable], list, tuple],
        num_epochs: int,
        num_epochs_warmup: int,
        adv_lambda: float,
        learning_rate: float,
        learning_rate_bottleneck: float,
        learning_rate_task_head: float,
        learning_rate_adv_head: float,
        optimizer_warmup_steps: int,
        max_grad_norm: float,
        output_dir: Union[str, os.PathLike],
        triplets: bool = False,
        protected_key: Optional[Union[str, list, tuple]] = None,
        checkpoint_name: Optional[str] = None,
        seed: Optional[int] = None,
        training_method: str = "par",
    ) -> None:

        if not isinstance(loss_fn_protected, (list, tuple)):
            loss_fn_protected = [loss_fn_protected]
        if not isinstance(pred_fn_protected, (list, tuple)):
            pred_fn_protected = [pred_fn_protected]
        if not isinstance(metrics_protected, (list, tuple)):
            metrics_protected = [metrics_protected]
        if not isinstance(protected_key, (list, tuple)):
            protected_key = [protected_key]
        if protected_key[0] is None:
            protected_key = list(range(len(protected_key)))

        self.global_step = 0
        num_epochs_total = num_epochs + num_epochs_warmup
        train_steps = len(train_loader) * num_epochs_total

        self._init_optimizer_and_schedule(
            train_steps,
            learning_rate,
            learning_rate_task_head,
            learning_rate_adv_head,
            learning_rate_bottleneck,
            optimizer_warmup_steps
        )

        self.zero_grad()
        # Parallel Training ( Each batch does task + attribute(s) )
        if training_method == "par":
            loss_fn_list = [loss_fn] + list(loss_fn_protected)
            pred_fn_list = [pred_fn] + list(pred_fn_protected)
            training_stages = self.training_stage
            train_str = "Epoch {}, {}"
            str_suffix = lambda d, suffix="": ", ".join([f"{k}{suffix}: {v}" for k,v in d.items()])

            train_iterator = trange(num_epochs_total, desc=train_str.format(0, "", ""), leave=False, position=0)

            for epoch in train_iterator:
                    if epoch < num_epochs_warmup:
                        _adv_lambda = 0.
                    else:
                        _adv_lambda = adv_lambda

                    if triplets:
                        self._step_triplets(
                            train_loader,
                            loss_fn,
                            logger,
                            max_grad_norm,
                            loss_fn_protected
                        )
                    else:
                        self._step(
                            train_loader,
                            loss_fn,
                            logger,
                            max_grad_norm,
                            loss_fn_protected,
                            _adv_lambda,
                            training_stages,
                            training_method
                        )

                    result = self.evaluate(
                        val_loader,
                        loss_fn_list,
                        pred_fn_list,
                        metrics,
                        training_stages,
                        training_method
                    )
                    result["lr"] = self.optimizer.param_groups[-1]["lr"]
                    logger.validation_loss(epoch, result)
                    if self.wandb_logger:
                        result["epochs"] = epoch
                        self.wandb_logger.log(result)

                    train_iterator.set_description(train_str.format(epoch, result), refresh=True)

                    cpt = self.save_checkpoint(Path(output_dir), checkpoint_name, seed)

        # Parallel Training (First Task is done fully then attribute(s) )
        elif training_method == "post":

            for label_idx, stage in enumerate(self.training_stage):
                # print(stage, label_idx)
                train_str = stage +" Epoch {}, {}"
                str_suffix = lambda d, suffix="": ", ".join([f"{k}{suffix}: {v}" for k, v in d.items()])
                train_iterator = trange(num_epochs_total, desc=train_str.format(0, "", ""), leave=False, position=0)
                loss_fn_list = [loss_fn] if stage == "task" else loss_fn_protected
                pred_fn_list = [pred_fn] if stage == "task" else pred_fn_protected
                if stage=="task":
                    task_early_stop = []
                    attr_early_stop = []
                    early_stop_counter = 0
                for epoch in train_iterator:
                    if epoch < num_epochs_warmup:
                        _adv_lambda = 0.
                    else:
                        _adv_lambda = adv_lambda

                    if triplets:
                        self._step_triplets(
                            train_loader,
                            loss_fn,
                            logger,
                            max_grad_norm,
                            loss_fn_protected
                        )

                    else:
                        self._step(
                            train_loader,
                            loss_fn,
                            logger,
                            max_grad_norm,
                            loss_fn_protected,
                            _adv_lambda,
                            [stage],
                            training_method
                        )

                    result = self.evaluate(
                        val_loader,
                        loss_fn_list,
                        pred_fn_list,
                        metrics,
                        [stage],
                        training_method
                    )
                    result["lr"] = self.optimizer.param_groups[-1]["lr"]
                    result["epochs"] = epoch
                    if stage == "task":
                        task_early_stop.append(result["task_loss"])
                        if task_early_stop[-1] == min(task_early_stop):
                            early_stop_counter = 0
                            print(f"Best Model saved at Epoch: {epoch} with loss: {task_early_stop[-1]}")
                            self.best_model = self.state_dict()
                        else:
                            early_stop_counter += 1
                            if early_stop_counter >= 5:
                                early_stop_counter = 0
                                self.load_state_dict(self.best_model)
                                cpt = self.save_checkpoint(Path(output_dir), checkpoint_name, seed)
                                logger.validation_loss(epoch, result)
                                if self.wandb_logger:
                                    result["epochs"] = epoch
                                    self.wandb_logger.log(result)

                                train_iterator.set_description(train_str.format(epoch, result), refresh=True)
                                break
                    else:
                        if stage != "age":
                            attr_early_stop.append(np.abs(result[f"{stage}_bacc"]-0.5))
                        else:
                            attr_early_stop.append(np.abs(result[f"{stage}_bacc"] - 0.2))
                        if epoch > 5:
                            if attr_early_stop[-1] == min(attr_early_stop):
                                print(f"Best Debiased model saved at Epoch: {epoch} with value {attr_early_stop[-1]}")
                                self.best_model = self.state_dict()
                                early_stop_counter = 0
                            else:
                                early_stop_counter += 1
                            if early_stop_counter >= 10:
                                cpt = self.save_checkpoint(Path(output_dir), checkpoint_name, seed)
                                logger.validation_loss(epoch, result)
                                if self.wandb_logger:
                                    result["epochs"] = epoch
                                    self.wandb_logger.log(result)

                                train_iterator.set_description(train_str.format(epoch, result), refresh=True)
                                break




                    logger.validation_loss(epoch, result)
                    if self.wandb_logger:
                        result["epochs"] = epoch
                        self.wandb_logger.log(result)

                    train_iterator.set_description(train_str.format(epoch, result), refresh=True)

                    cpt = self.save_checkpoint(Path(output_dir), checkpoint_name, seed)

            self.load_state_dict(self.best_model)

        else:
            print("WARNING TRAINING METHOD NOT RECOGNIZED")
            cpt = None

        cpt = self.save_checkpoint(Path(output_dir), checkpoint_name, seed)
        return cpt

    def evaluate_proxy_model(self,
                            val_loader,
                            stage):

        w = dict(zip(self.congater_names, [0 for _ in self.congater_names])) if self.congater_names else 0
        if stage != "task":
            w[f"{stage}"] = 1

        try:
            dev = self.device
        except AttributeError:
            dev = next(self.parameters()).device

        first_batch = True
        for batch in val_loader:
            inputs, labels = batch[0], batch[2]
            if isinstance(inputs, dict):
                inputs = dict_to_device(inputs, dev)
            else:
                inputs = inputs.to(dev)
            embeds = self._forward(w, **inputs)
            if first_batch:
                all_embeds = embeds.detach().cpu().numpy()
                all_labels = labels.detach().cpu().numpy()
                first_batch = False
            else:
                all_embeds = np.concatenate((all_embeds, embeds.detach().cpu().numpy()), axis=0)
                all_labels = np.concatenate((all_labels, labels.detach().cpu().numpy()))

        y_hat = self.proxy_model.predict(all_embeds)
        bc = balanced_accuracy_score(all_labels, y_hat)

        return 2 * (1. - 2 * (1-bc))

    #  Calculate Performance of the model based on different omega values
    @torch.no_grad()
    def w_performance(
            self,
            val_loader: DataLoader,
            test_loader: DataLoader,
            loss_fn: list[Callable],
            pred_fn: list[Callable],
            metrics: Dict[str, Callable],
            w,
            interpolate=False,

    ) -> dict:
        if type(w) is int or type(w) is float:
            w = [w]
        result = {}
        w = dict(zip(self.congater_names, [value for value in w])) if self.congater_names else {"task": 0}
        print("Interpolate", interpolate)
        forward_fn = lambda x: self.forward(w=w, interpolate=interpolate, **x)
        desc = ''.join([f"{k}_{v}" for k, v in w.items()])
        val_res = congater_evaluate_model(
            self,
            val_loader,
            loss_fn[0],
            pred_fn[0],
            metrics,
            label_idx = 1,
            desc=desc,
            forward_fn=forward_fn,
            stage="task",
            w=w,
            interpolate=interpolate)
        test_res = congater_evaluate_model(
            self,
            test_loader,
            loss_fn[0],
            pred_fn[0],
            metrics,
            label_idx = 1,
            desc=desc,
            forward_fn=forward_fn,
            stage="task",
            w=w,
            interpolate=interpolate)
        for k, v in w.items():
            result[f"w_{k}"] = v
        for k, v in val_res.items():
            result[f"w_eval_{k}"] = v
        for k, v in test_res.items():
            result[f"w_test_{k}"] = v
        if self.wandb_logger:
            self.wandb_logger.log(result)
        return result

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        loss_fn: list[Callable],
        pred_fn: list[Callable],
        metrics: Dict[str, Callable],
        training_stages: list = ["task"],
        training_method: str = "par",

    ) -> dict:


        result = {}
        for label_idx, stage in enumerate(training_stages):
            desc = f"{stage}"
            label_idx = self.training_stage.index(stage)
            lbl_num = label_idx + 1
            w = dict(zip(self.congater_names, [0 for _ in self.congater_names])) if self.congater_names else 0

            if stage == "task":
                forward_fn = lambda x: self.forward(w=w, **x)
                ls_index = label_idx
            elif stage != "task":
                w[f"{stage}"] = 1
                if len(training_stages) == 1:
                    ls_index = label_idx - 1
                else:
                    ls_index = label_idx
                forward_fn = lambda x: self.forward_protected(head_idx=label_idx - 1, w=w, **x)

            res = congater_evaluate_model(
                self,
                val_loader,
                loss_fn[ls_index],
                pred_fn[ls_index],
                metrics,
                label_idx=lbl_num,
                desc=desc,
                forward_fn=forward_fn,
                stage=stage,
                w=w)
            result.update(res)


        return result

    def _step(
        self,
        train_loader: DataLoader,
        loss_fn: list[Callable],
        logger: TrainLogger,
        max_grad_norm: float,
        loss_fn_protected: Union[list, tuple],
        adv_lambda: float,
        training_stages: list,
        training_method: str,
    ) -> None:
        self.train()
        epoch_str = "training - step {}, loss: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan), leave=False, position=1)
        # first_batch = True
        for step, batch in enumerate(epoch_iterator):
            loss_desc = {}
            for idx, stage in enumerate(training_stages):
                w = dict(zip(self.congater_names, [0 for _ in self.congater_names])) if self.congater_names else 0
                if stage == "task":
                    self.set_trainable_parameters(self.default_trainable_parameters)
                else:
                    if training_method == "post":
                        self.set_trainable_parameters(f"{stage}+adv_head")
                        idx = 0 if stage != "age" else 1
                    elif training_method == "par":
                        idx = idx - 1
                        self.set_trainable_parameters(f"{stage}+adv_head.{idx}")
                    w[f"{stage}"] = 1

                loss = 0.
                inputs, labels_task = batch[:2]
                labels_protected = batch[2:]
                inputs = dict_to_device(inputs, self.device)
                hidden = self._forward(w, **inputs)
                outputs_task = self.task_head(hidden)
                loss_task = loss_fn(outputs_task, labels_task.to(self.device))
                #TODO: Implement Normalized ADV, Better Gradient Flow without change in the learning rate
                # loss += (1-adv_lambda)*loss_task
                loss += loss_task
                if stage != "task" and self.congater_names:

                    outputs_protected = self.adv_head[idx].forward_reverse(hidden, lmbda=adv_lambda)
                    loss_protected = get_mean_loss(outputs_protected,
                                                   labels_protected[idx].to(self.device), loss_fn_protected[idx])
                    loss += loss_protected

                else:
                   loss_protected = torch.zeros_like(loss).detach()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                self.optimizer.step()
                if not self.no_scheduler:
                    self.scheduler.step()
                self.zero_grad()
                loss_desc[f"{stage}"] = loss.item()
                losses_dict = {
                    f"{stage} train total_adv": loss.item(),
                    f"{stage} train task_adv": loss_task.item(),
                    f"{stage} train protected": loss_protected.item(),
                    "step": self.global_step
                }
                # if self.wandb_logger:
                #     self.wandb_logger.log(losses_dict)
                logger.step_loss(self.global_step, losses_dict)
                # inputs, labels = batch[0], batch[2]
            # exit()
            # if first_batch:
            #     all_embeds = hidden.detach().cpu().numpy()
            #     all_labels = labels.detach().cpu().numpy()
            #     first_batch = False
            # else:
            #     all_embeds = np.concatenate((all_embeds, hidden.detach().cpu().numpy()), axis=0)
            #     all_labels = np.concatenate((all_labels, labels.detach().cpu().numpy()))
            epoch_iterator.set_description(f"{step} - {loss_desc}", refresh=True)

        self.global_step += 1
        # self.proxy_model = svm.SVC(C=1, probability=True, max_iter=300)
        # self.proxy_model.fit(all_embeds, all_labels)

    def _step_triplets(
        self,
        train_loader: DataLoader,
        loss_fn: Callable,
        logger: TrainLogger,
        max_grad_norm: float,
        loss_fn_protected: Union[list, tuple]
    ) -> None:
        self.train()

        epoch_str = "training - step {}, loss: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan), leave=False, position=1)
        for step, batch in enumerate(epoch_iterator):

            loss = 0.

            inputs, neg, pos, weights, labels_task = batch[:5]
            labels_protected = batch[5:]

            inputs = dict_to_device(inputs, self.device)
            neg = dict_to_device(neg, self.device)
            pos = dict_to_device(pos, self.device)

            hidden = self._forward(**inputs)
            outputs_task = self.task_head(hidden)
            loss_task = loss_fn(outputs_task, labels_task.to(self.device))
            loss += loss_task

            hidden_pos = self._forward(**pos)
            hidden_neg = self._forward(**neg)
            loss_triplets = triplet_margin_loss(hidden, hidden_pos, hidden_neg, margin=0, reduction="none")
            loss_triplets = (loss_triplets * weights.to(self.device)).mean()
            loss += (loss_triplets * len(labels_protected))

            for i, (l, loss_fn_prot) in enumerate(zip(labels_protected, loss_fn_protected)):
                outputs_protected = self.adv_head[i](hidden.detach())
                loss_adv_head = get_mean_loss(outputs_protected, l.to(self.device), loss_fn_prot)
                loss_adv_head.backward()
                
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

            self.optimizer.step()
            if not self.no_scheduler:
                self.scheduler.step()
            self.zero_grad()

            losses_dict = {
                "total_adv": loss.item(),
                "task_adv": loss_task.item(),
                "protected": loss_adv_head.item(),
                "triplets": loss_triplets.item()
            }
            logger.step_loss(self.global_step, losses_dict)

            epoch_iterator.set_description(epoch_str.format(step, loss.item()), refresh=True)

            self.global_step += 1


    def _init_optimizer_and_schedule(
        self,
        num_training_steps: int,
        learning_rate: float,
        learning_rate_task_head: float,
        learning_rate_adv_head: float,
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
                "lr": learning_rate_task_head
            },
            {
                "params": self.adv_head.parameters(),
                "lr": learning_rate_adv_head
            }]

        if self.congater_names:
            print("Added Congater Parameters to Optimizer")
            optimizer_params.append({
                "params": self.congater.parameters(),
                "lr": learning_rate_bottleneck
            })

        self.optimizer = AdamW(optimizer_params, betas=(0.9, 0.999), eps=1e-08)
        if not self.no_scheduler:
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                             num_warmup_steps=num_warmup_steps,
                                                             num_training_steps=num_training_steps*(3 if len(self.congater_names)>1 else 2)
            )


    def make_checkpoint_name(
        self,
        seed: Optional[int] = None
    ):
        filename_parts = [
            self.model_name.split('/')[-1],
            "congater",
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
            "congater_names": self.congater_names,
            "num_gate_layers": self.num_gate_layers,
            "gate_squeeze_ratio": self.gate_squeeze_ratio,
            "congater_position": self.congater_position,
            "num_labels_task": self.num_labels_task,
            "num_labels_protected": self.num_labels_protected,
            "task_dropout": self.task_dropout,
            "task_n_hidden": self.task_n_hidden,
            "adv_dropout": self.adv_dropout,
            "adv_n_hidden": self.adv_n_hidden,
            "adv_count": self.adv_count,
            "bottleneck": self.has_bottleneck,
            "bottleneck_dim": self.bottleneck_dim,
            "bottleneck_dropout": self.bottleneck_dropout,
            "encoder_state_dict": self.encoder_module.state_dict(),
            "congater_state_dict": self.congater.state_dict(),
            "bottleneck_state_dict": self.bottleneck.state_dict(),
            "task_head_state_dict": self.task_head.state_dict(),
            "adv_head_state_dict": self.adv_head.state_dict()
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
            model_name=info_dict['model_name'],
            congater_names=info_dict['congater_names'],
            num_gate_layers=info_dict['num_gate_layers'],
            gate_squeeze_ratio=info_dict['gate_squeeze_ratio'],
            congater_position=info_dict['congater_position'],
            num_labels_task=info_dict['num_labels_task'],
            num_labels_protected=info_dict['num_labels_protected'],
            task_dropout=info_dict['task_dropout'],
            task_n_hidden=info_dict['task_n_hidden'],
            adv_dropout=info_dict['adv_dropout'],
            adv_n_hidden=info_dict['adv_n_hidden'],
            adv_count=info_dict['adv_count'],
            bottleneck=info_dict['bottleneck'],
            bottleneck_dim=info_dict['bottleneck_dim'],
            bottleneck_dropout=info_dict['bottleneck_dropout']
        )
        # print(info_dict.keys())

        cls_instance.bert.load_state_dict(info_dict['encoder_state_dict'])
        cls_instance.bottleneck.load_state_dict(info_dict['bottleneck_state_dict'])
        cls_instance.adv_head.load_state_dict(info_dict['adv_head_state_dict'])
        cls_instance.task_head.load_state_dict(info_dict['task_head_state_dict'])
        cls_instance.congater.load_state_dict(info_dict['congater_state_dict'])

        cls_instance.eval()

        return cls_instance
