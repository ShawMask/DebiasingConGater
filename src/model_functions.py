import math
from pathlib import Path
from copy import deepcopy
# from tqdm.notebook import tqdm, trange
from tqdm import tqdm, trange
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.models.model_base import BaseModel
from src.models.model_adv import AdvModel
from src.models.model_task import TaskModel
from src.training_logger import TrainLogger
from src.utils import dict_to_device, get_param_from_name, evaluate_model

from typing import Optional, Union, Callable, Dict


AVAILABLE_MODEL_CLASSES = [
    TaskModel,
    AdvModel,
]


@torch.no_grad()
def merge_models(
    *model_list, mean: bool = False, mean_ignore_zero: bool = False
) -> torch.nn.Module:
    # assert all weights match
    sets = [set([n for n, _ in m.named_parameters()]) for m in model_list]
    try:
        intersect = sets[0].intersection(*sets[1:])
        assert len(sets[0]) == len(intersect)
    except:
        all_keys = sets[0].union(*sets[1:])
        missing = [k for k in all_keys if k not in intersect]
        raise Exception(f"Keys {missing} not present in all models")

    if mean and mean_ignore_zero:
        norm_dict = {}
        for m in model_list:
            for p_name, p in m.named_parameters():
                try:
                    norm_dict[p_name] += (p!=0.)
                except:
                    norm_dict[p_name] = (p!=0.).long()
        norm_dict = {p_name: v.clamp(min=1) for p_name, v in norm_dict.items()}

    model_frame = deepcopy(model_list[0])
    for p_name, p in model_frame.named_parameters():
        p.zero_()
        for i in range(len(model_list)):
            p_add = get_param_from_name(model_list[i], p_name)
            if mean:
                if mean_ignore_zero:
                    p_add /= norm_dict[p_name]
                else:
                    p_add /= len(model_list)
            p += p_add

    return model_frame


@torch.no_grad()
def merge_adv_models(
    *adv_model_list, base_model: BaseModel = None, mean_diff_weights: bool = False, mean_ignore_zero: bool = False
):
    diff_weights = [m.get_diff_weights(0, as_module=True) for m in adv_model_list]
    if mean_diff_weights and len(diff_weights)>1:
        diff_weights = [merge_models(*diff_weights, mean=True, mean_ignore_zero=mean_ignore_zero)]

    if base_model is None:
        base_weights = adv_model_list[0].get_base_weights(as_module=True)
    else:
        base_weights = base_model.encoder

    return merge_models(base_weights, *diff_weights, mean=False)



@torch.no_grad()
def merge_modular_model(
    modular_model, mean_diff_weights: bool = False, mean_ignore_zero: bool = False
):
    diff_weights = []
    for i in range(modular_model.sparse_task, modular_model.n_parametrizations):
        diff_weights.append(
            modular_model.get_diff_weights(i, as_module=True)
        )
        
    if mean_diff_weights and len(diff_weights)>1:
        diff_weights = [merge_models(*diff_weights, mean=True, mean_ignore_zero=mean_ignore_zero)]

    if modular_model.sparse_task:
        base_weights = modular_model.get_diff_weights(0, as_module=True)
    else:
        base_weights = modular_model.get_base_weights(as_module=True)

    return merge_models(base_weights, *diff_weights, mean=False)


@torch.no_grad()
def generate_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    forward_fn: Callable,  # = lambda m, x: m(**x)
):
    device = next(model.parameters()).device
    model.eval()
    emb_list = []
    labels_list = []
    for batch in tqdm(loader, desc="generating embeddings"):
        inputs = batch[0]
        inputs = dict_to_device(inputs, device)
        emb = forward_fn(model, inputs)
        emb_list.append(emb.cpu())
        labels_list.append(batch[1:])
    labels = [torch.cat(x) for x in zip(*labels_list)]
    embeddings = torch.cat(emb_list)
    return embeddings, *labels


def train_head(
    head: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    val_loader: DataLoader,
    logger: TrainLogger,
    loss_fn: Callable,
    pred_fn: Callable,
    metrics: Dict[str, Callable],
    optim: Optimizer,
    weight_decay: float,
    num_epochs: int,
    lr: float,
    cooldown: int = 5,
    device: Optional[Union[str, torch.device]] = None,
    desc: str = "",
    wandb_logger: object = None,
):

    logger.reset()

    if device:
        head.to(device)
    else:
        device = next(head.parameters()).device

    optimizer = optim(head.parameters(), lr=lr, weight_decay=weight_decay)

    global_step = 0
    train_str = "Epoch {}, {}"
    result_str = lambda x: ", ".join([f"{k}: {v}" for k,v in x.items()])

    performance_decrease_counter = 0
    train_iterator = trange(num_epochs, desc=train_str.format(0, ""), leave=False, position=0)
    probe_results = {}
    best_bacc = []
    for epoch in train_iterator:

        epoch_str = "training {} - step {}, loss: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(desc, 0, math.nan), leave=False, position=1)

        head.train()
        loss_list = []

        for step, (inputs, labels) in enumerate(epoch_iterator):
            outputs = head(inputs.to(device))
            loss = loss_fn(outputs, labels.to(device))

            loss.backward()
            optimizer.step()
            # scheduler.step()
            head.zero_grad()
            loss_list.append(loss.item())
            logger.step_loss(global_step, loss.item(), lr=lr, suffix=desc)
            epoch_iterator.set_description(epoch_str.format(desc, step, loss.item()), refresh=True)

        global_step += 1 ### Previouly inside the loop

        result = evaluate_model(head, val_loader, loss_fn, pred_fn, metrics)
        test_result = evaluate_model(head, test_loader, loss_fn, pred_fn, metrics)
        if "adp_inter" in desc:
            result = {k.split("_")[0]:v for k,v in result.items()}
            test_result = {k.split("_")[0]: v for k, v in test_result.items()}
            best_bacc.append(test_result["bacc"])
        if wandb_logger:
            wandb_logger.log({f"probe_epochs": global_step, f"eval_{desc}": result, f"test_{desc}": test_result})
        logger.validation_loss(epoch, result, desc)
        logger.test_loss(epoch, test_result, desc)

        train_iterator.set_description(
            train_str.format(epoch, result_str(result)), refresh=True
        )

        if logger.is_best(result["bacc"], ascending=True):
            best_result = result
            best_test_result = test_result
            best_epoch = epoch
            performance_decrease_counter = 0
        else:
            performance_decrease_counter += 1


    print("logger desc:",  desc)

    if ("_g_" in desc) or ("_d_" in desc):
        log_name = desc.split("_")[0:]
        for i in range(2, len(log_name), 2):
            probe_results[f"probe_w_{log_name[i]}"] = float(log_name[i+1])
        for k, v in result.items():
            probe_results[f"probe_final_eval_{k}_{log_name[1]}"] = v
        for k, v in test_result.items():
            probe_results[f"probe_final_test_{k}_{log_name[1]}"] = v

        # print(f"debug {probe_results}")
        if wandb_logger:
            wandb_logger.log(probe_results)
    elif "adp_inter" in desc:
        log_name = desc.split("_")[0:]

        probe_results[f"probe_w_{log_name[-2][0]}"] = float(log_name[-1])
        for k, v in result.items():
            probe_results[f"probe_final_eval2_{log_name[-2]}_{k}"] = v
        for k, v in test_result.items():
            probe_results[f"probe_final_test_{log_name[-2]}_{k}"] = v
        probe_results[f"probe_final_eval_bacc_{log_name[-2]}"] = max(best_bacc)

        if wandb_logger:
            wandb_logger.log(probe_results)
    else:
        if wandb_logger:
            res_, rest_, best_ = {}, {}, {}
            for key, value in result.items():
                res_[f"probe_final_eval_{key}"] = value
            for key, value in test_result.items():
                res_[f"probe_final_test_{key}"] = value
            for key, value in best_test_result.items():
                res_[f"probe_best_test_{key}"] = value
            wandb_logger.log(res_)


    prefix = desc + ': ' if desc else ''
    print(f"{prefix}Final Eval result after " + train_str.format(epoch, result_str(result)))
    print(f"{prefix}Final Test result after " + train_str.format(epoch, result_str(test_result)))


    return test_result


def model_factory(
    cp_path: Union[str, Path],
    map_location: Union[str, torch.device] = torch.device('cpu'),
    **kwargs
    ):
    info_dict = torch.load(cp_path, map_location=map_location)
    model_cls = eval(info_dict["cls_name"])
    assert model_cls in AVAILABLE_MODEL_CLASSES, \
        f"Model Class {model_cls} is not in available model classes: {', '.join(AVAILABLE_MODEL_CLASSES)}"
    load_cp_args = model_cls.load_checkpoint.__code__.co_varnames
    model_cls_kwargs = {k:v for k,v in kwargs.items() if k in load_cp_args}
    return model_cls.load_checkpoint(cp_path, map_location=map_location, **model_cls_kwargs)

