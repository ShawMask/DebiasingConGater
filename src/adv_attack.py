import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup

from src.models.model_base import BaseModel
from src.models.model_heads import AdvHead
from src.training_logger import TrainLogger
from src.model_functions import train_head, generate_embeddings
from src.data_handler import get_data
from src.utils import get_callables, get_mean_loss

from typing import Callable, Dict, Optional, Union

@torch.no_grad()
def get_hidden_dataloader(
    trainer: BaseModel,
    loader: DataLoader,
    label_idx: int,
    shuffle: bool = True,
    batch_size: Optional[int] = None,
    congater = 0,
):
    if congater:
        data = generate_embeddings(trainer, loader, forward_fn=lambda m, x: m._forward(m.evaluate_w, **x))
    else:
        data = generate_embeddings(trainer, loader, forward_fn = lambda m, x: m._forward(**x))
    bs = loader.batch_size if batch_size is None else batch_size
    ds = TensorDataset(data[0], data[label_idx])
    return DataLoader(ds, shuffle=shuffle, batch_size=bs, drop_last=False)


def adv_attack(
    train_loader: DataLoader,
    test_loader: DataLoader,
    val_loader: DataLoader,
    logger: TrainLogger,
    loss_fn: Callable,
    pred_fn: Callable,
    metrics: Dict[str, Callable],
    num_labels: int,
    adv_n_hidden: int,
    adv_count: int,
    adv_dropout: int,
    weight_decay: float,
    num_epochs: int,
    lr: float,
    cooldown: int = 5,
    create_hidden_dataloader: bool = True,
    trainer: Optional[BaseModel] = None,
    batch_size: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    label_idx: int = 2,
    logger_suffix: str = "adv_attack",
    congater: int = 0,
    wandb_logger: object = None,
):

    if trainer is None:
        input_size = next(iter(val_loader))[0].shape[1]
    else:
        input_size = trainer.in_size_heads

    if device is None:
        try:
            device = next(trainer.parameters()).device
        except AttributeError:
            print("device and trainer is None, defaulting to cpu")
            device = torch.device("cpu")

    adv_head = AdvHead(
        adv_count,
        hid_sizes=[input_size]*(adv_n_hidden+1),
        num_labels=num_labels,
        dropout=adv_dropout
    )
    adv_head.to(device)

    if create_hidden_dataloader:
        assert isinstance(trainer, BaseModel), "if create_hidden_dataloader=True, trainer needs to be defined as BaseModel class"
        assert batch_size is not None, "if create_hidden_dataloader=True, batch size needs to be defined"
        train_loader = get_hidden_dataloader(trainer, train_loader, label_idx=label_idx,
                                             batch_size=batch_size, congater=congater)
        val_loader = get_hidden_dataloader(trainer, val_loader, label_idx=label_idx, shuffle=False,
                                           batch_size=batch_size, congater=congater)
        test_loader = get_hidden_dataloader(trainer, test_loader, label_idx=label_idx, shuffle=False,
                                            batch_size=batch_size, congater=congater)

    mean_loss_fn = lambda outputs, labels: get_mean_loss(outputs, labels, loss_fn)

    test_result = train_head(
        head = adv_head,
        train_loader = train_loader,
        test_loader = test_loader,
        val_loader = val_loader,
        logger = logger,
        loss_fn = mean_loss_fn,
        pred_fn = pred_fn,
        metrics = metrics,
        optim = AdamW,
        weight_decay = weight_decay,
        num_epochs = num_epochs,
        lr = lr,
        cooldown = cooldown,
        desc = logger_suffix,
        wandb_logger = wandb_logger
    )

    return test_result


def run_adv_attack(
    base_args,
    args_train,
    args_attack,
    trainer,
    train_logger,
    wandb_logger,
    congater_interpolate=False,
):

    train_loader,  test_loader, val_loader, _, num_labels_protected, protected_key, protected_class_weights = get_data(
        args_train,
        use_all_attr = True,
        compute_class_weights = args_train.weighted_loss_protected,
        device = trainer.device,
        debug = base_args.debug
    )

    if "modular" in base_args.model_type:
        train_data = []
        val_data = []
        for i in range(trainer.n_embeddings):
            trainer.set_debiased(bool(i), debiased_par_idx=max(0, i-1))
            train_data.append(generate_embeddings(trainer, train_loader, forward_fn = lambda m, x: m._forward(**x)))
            val_data.append(generate_embeddings(trainer, val_loader, forward_fn = lambda m, x: m._forward(**x)))

    elif "congater" in base_args.model_type:
            train_data = generate_embeddings(trainer, train_loader,
                                             forward_fn=lambda m, x: m._forward(m.evaluate_w, interpolate=congater_interpolate, **x))
            val_data = generate_embeddings(trainer, val_loader,
                                           forward_fn=lambda m, x: m._forward(m.evaluate_w, interpolate=congater_interpolate, **x))
            test_data = generate_embeddings(trainer, test_loader,
                                            forward_fn=lambda m, x: m._forward(m.evaluate_w, interpolate=congater_interpolate, **x))

    else:
        train_data = generate_embeddings(trainer, train_loader, forward_fn = lambda m, x: m._forward(**x))
        val_data = generate_embeddings(trainer, val_loader, forward_fn = lambda m, x: m._forward(**x))
        test_data = generate_embeddings(trainer, test_loader, forward_fn = lambda m, x: m._forward(**x))


    device = trainer.device

    # free up memory
    # del trainer
    adv_result_list = {}
    for (i, (num_lbl_prot, prot_k, prot_w)) in enumerate(zip(num_labels_protected, protected_key, protected_class_weights)):
        label_idx = i+2
        loss_fn, pred_fn, metrics = get_callables(num_lbl_prot, class_weights = prot_w)


        train_loader = DataLoader(
            TensorDataset(train_data[0], train_data[label_idx]), shuffle=True, batch_size=args_attack.attack_batch_size, drop_last=False
        )
        val_loader = DataLoader(
            TensorDataset(val_data[0], val_data[label_idx]), shuffle=False, batch_size=args_attack.attack_batch_size, drop_last=False
        )
        test_loader = DataLoader(
            TensorDataset(test_data[0], test_data[label_idx]), shuffle=False, batch_size=args_attack.attack_batch_size, drop_last=False
        )

        emb_name = base_args.model_type if "congater" not in base_args.model_type else ""

        if base_args.prot_key_idx is None:
            if "congater" in base_args.model_type:
                str_w = "_".join([f"{k[0]}_{v}" for k, v in trainer.evaluate_w.items()])
            else:
                emb_name = emb_name+"_all"
        else:
            emb_name = emb_name+f"_{protected_key[base_args.prot_key_idx]}"
        if "congater" in base_args.model_type:
            logger_suffix = "_".join(["probe", emb_name, f"{prot_k}_{trainer.evaluate_w[prot_k]}"]) \
                if "congater" not in base_args.model_type else "_".join(["probe", prot_k, str_w])
        else:
            logger_suffix = "_".join(["probe", prot_k, emb_name])
        adv_result = adv_attack(
            train_loader = train_loader,
            test_loader = test_loader,
            val_loader = val_loader,
            logger = train_logger,
            loss_fn = loss_fn,
            pred_fn = pred_fn,
            metrics = metrics,
            num_labels = num_lbl_prot,
            adv_n_hidden = args_attack.adv_n_hidden,
            adv_count = args_attack.adv_count,
            adv_dropout = args_attack.adv_dropout,
            weight_decay=args_attack.weight_decay,
            num_epochs = args_attack.num_epochs,
            lr = args_attack.learning_rate,
            cooldown = args_attack.cooldown,
            create_hidden_dataloader = False,
            device = device,
            logger_suffix = logger_suffix,
            wandb_logger = wandb_logger,
        )
        adv_result_list[prot_k] = adv_result

    return adv_result_list