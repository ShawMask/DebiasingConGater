import os
import math
from enum import Enum, auto
from tqdm import tqdm
import contextlib
import torch
from torch.utils.data import DataLoader
import torch.nn.utils.parametrize as parametrize
from transformers import AutoModel, PretrainedConfig

from typing import Union, List, Tuple, Optional, Dict, Callable

from src.models.weight_parametrizations import DiffWeightFinetune, DiffWeightFixmask
from src.utils import dict_to_device

class ModelState(Enum):
    INIT = auto()
    FINETUNING = auto()
    FIXMASK = auto()


class BertModel(torch.nn.Module):

    @property
    def encoder_module(self) -> PretrainedConfig:
        if isinstance(self.bert, torch.nn.DataParallel):
            return self.bert.module
        else:
            return self.bert

    @property
    def device(self) -> torch.device:
        return next(self.bert.parameters()).device

    @property
    def model_type(self) -> str:
        return self.bert.config.model_type

    @property
    def model_name(self) -> str:
        return self.bert.config._name_or_path

    @property
    def hidden_size(self) -> int:
        return self.bert.embeddings.word_embeddings.embedding_dim

    @property
    def total_layers(self) -> int:
        possible_keys = ["num_hidden_layers", "n_layer"]
        cfg = self.bert.config
        for k in possible_keys:
            if k in cfg.__dict__:
                return getattr(cfg, k) + 1 # +1 for embedding layer and last layer
        raise Exception("number of layers of pre trained model could not be determined")

    def __init__(self, model_name: str, **kwargs):
        super().__init__()
        kwargs = {}
        self.bert = AutoModel.from_pretrained(model_name, **kwargs)
        # if "roberta" in model_name:
        #     self.bert = self.bert.roberta

    def _forward(self, **x) -> torch.Tensor:
        # embed = self.bert.embeddings(x["input_ids"], x["token_type_ids"])
        # x = {"hidden_states": embed, "encoder_attention_mask": x["attention_mask"]}
        # for i in range(len(self.bert.encoder.layer)):
        #     x["hidden_states"] = self.bert.encoder.layer[i](**x)[0]
        return self.bert(**x)[0][:, 0] ### Why Basline is Returning this instead of Pooler?
        # return self.bert.pooler(x["hidden_states"])

    @torch.no_grad()
    def _evaluate(
        self,
        val_loader: DataLoader,
        forward_fn: Callable,
        loss_fn: Callable,
        pred_fn: Callable,
        metrics: Dict[str, Callable],
        label_idx: int = 1,
        desc: str = "",
        **kwargs
        ) -> dict:

        self.eval()

        eval_loss = 0.
        output_list = []
        val_iterator = tqdm(val_loader, desc=f"evaluating {desc}", leave=False, position=1)
        for i, batch in enumerate(val_iterator):

            inputs, labels = batch[0], batch[label_idx]
            if isinstance(inputs, dict):
                inputs = dict_to_device(inputs, self.device)
            else:
                inputs = inputs.to(self.device)
            logits = forward_fn(inputs, **kwargs)
            if isinstance(logits, list):
                eval_loss += torch.stack([loss_fn(x.cpu(), labels) for x in logits]).mean().item()
                preds, _ = torch.mode(torch.stack([pred_fn(x.cpu()) for x in logits]), dim=0)
            else:
                eval_loss += loss_fn(logits.cpu(), labels).item()
                preds = pred_fn(logits.cpu())
            output_list.append((
                preds,
                labels
            ))

        p, l = list(zip(*output_list))
        predictions = torch.cat(p, dim=0)
        labels = torch.cat(l, dim=0)

        result = {metric_name: metric(predictions, labels) for metric_name, metric in metrics.items()}
        result["loss"] = eval_loss / (i+1)

        return result

    def _get_mean_loss(self, outputs: Union[torch.Tensor, List[torch.Tensor]], labels: torch.Tensor, loss_fn: Callable) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
        losses = []
        for output in outputs:
            losses.append(loss_fn(output, labels))
        return torch.stack(losses).mean()        

    def forward(self, **x) -> torch.Tensor:
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def _step(self, *args, **kwargs):
        raise NotImplementedError

    def _init_optimizer_and_schedule(self, *args, **kwargs):
        raise NotImplementedError

    def save_checkpoint(self, output_dir: Union[str, os.PathLike], *args, **kwargs) -> None:
        raise NotImplementedError

    @classmethod
    def load_checkpoint(cls, filepath: Union[str, os.PathLike], *args, **kwargs) -> torch.nn.Module:
        raise NotImplementedError

    def to(self, device: Union[list, Union[str, torch.device]], *args, **kwargs) -> None:
        if isinstance(device, list):
            asssert_fn = lambda x: x=="cuda" if isinstance(x, str) else x.type=="cuda"
            assert all([asssert_fn(d) for d in device]), "if list of devices is given, all must be of type 'cuda'"
            super().to(device[0])
            if len(device)>1:
                self.bert = torch.nn.DataParallel(self.bert, device_ids=device)
        else:
            self._remove_parallel()
            super().to(device)

    def cpu(self):
        self._remove_parallel()
        super().cpu()

    def cuda(self, *args, **kwargs) -> None:
        self._remove_parallel()
        super().cuda(*args, **kwargs)

    def _remove_parallel(self) -> None:
        if isinstance(self.bert, torch.nn.DataParallel):
            self.bert = self.bert.module

