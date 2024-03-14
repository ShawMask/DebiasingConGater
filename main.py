import argparse
import ruamel.yaml as yaml
import torch

from transformers import logging
import warnings
import wandb
# Version deprecated warning removal
warnings.filterwarnings('ignore')
from itertools import product
from torchsummary import summary
# Transformer Warning Remover
logging.set_verbosity_error()
import os




from src.models.model_adv import AdvModel
from src.models.adp_debias import AdvAdapter
from src.models.model_congater_adv import ConGaterModel
from src.models.model_task import TaskModel
from src.models.task_adapter import TaskAdapter
from src.model_functions import model_factory
from src.adv_attack import run_adv_attack
from src.data_handler import get_data, get_glu_data
from src.utils import (
    get_device,
    set_num_epochs_debug,
    set_dir_debug,
    get_logger,
    get_callables_wrapper,
    set_optional_args
)

# (1) Start ------training function for # Baseline, BaselineADV, Adapter, AdapterADV and ConGater-----
def train_baseline_task(
    device,
    train_loader,
    test_loader,
    val_loader,
    num_labels,
    loss_fn,
    pred_fn,
    metrics,
    train_logger,
    args_train,
    encoder_state_dict = None,
    seed = None,
    log_wandb = 0,
):
    wandb_log_path = os.path.join(train_logger.log_dir, train_logger.logger_name)
    tag = str(train_logger.log_dir).split("_")[-1]

    project_name = f"{args_train.wandb_project}_{tag}"
    args_train.experiment_name = train_logger.logger_name

    wandb_logger = wandb.init(dir=wandb_log_path, project=project_name,
                              name=train_logger.logger_name, config=args_train, tags=[tag]) if log_wandb else None

    trainer = TaskModel(
        model_name = args_train.model_name,
        num_labels = num_labels,
        dropout = args_train.task_dropout,
        n_hidden = args_train.task_n_hidden,
        bottleneck = args_train.bottleneck,
        bottleneck_dim = args_train.bottleneck_dim,
        bottleneck_dropout = args_train.bottleneck_dropout,
        encoder_state_dict = encoder_state_dict,
        wandb_logger = wandb_logger,
        augment=args_train.augment,
        no_scheduler=args_train.no_scheduler,
    )

    trainer.to(device)
    print(trainer.param_spec())

    trainer_cp = trainer.fit(
        train_loader = train_loader,
        val_loader = val_loader,
        logger = train_logger,
        loss_fn = loss_fn,
        pred_fn = pred_fn,
        metrics = metrics,
        num_epochs = args_train.num_epochs,
        learning_rate = args_train.learning_rate,
        learning_rate_bottleneck = args_train.learning_rate_bottleneck,
        learning_rate_head = args_train.learning_rate_task_head,
        optimizer_warmup_steps = args_train.optimizer_warmup_steps,
        max_grad_norm = args_train.max_grad_norm,
        output_dir = args_train.output_dir,
        cooldown = args_train.cooldown,
        checkpoint_name = train_logger.logger_name + ".pt",
        seed = seed
    )

    trainer = TaskModel.load_checkpoint(trainer_cp)
    trainer.wandb_logger = wandb_logger
    trainer.to(device)

    if args_train.ds in ["hatespeech", "bios", "pan16"]:
        test_result= trainer.evaluate(
                        test_loader,
                        loss_fn,
                        pred_fn,
                        metrics
                    )
        test_str = {k: v for k, v in test_result.items()}
        if trainer.wandb_logger:
            wandb_test_str = {f"test task {k}": v for k, v in test_result.items()}
            trainer.wandb_logger.log(wandb_test_str)
        train_logger.writer.add_scalar("test/acc_task", test_str['acc'])
        train_logger.writer.add_scalar("test/bacc_task", test_str['bacc'])
        print('{:10}'.format("Final results test"))
        print(test_str)

    return trainer


def train_adapter_task(
    device,
    train_loader,
    test_loader,
    val_loader,
    num_labels,
    loss_fn,
    pred_fn,
    metrics,
    train_logger,
    args_train,
    encoder_state_dict = None,
    seed = None,
    log_wandb = 0,
):
    wandb_log_path = os.path.join(train_logger.log_dir, train_logger.logger_name)
    tag = str(train_logger.log_dir).split("_")[-1]

    project_name = f"{args_train.wandb_project}_{tag}"
    args_train.experiment_name = train_logger.logger_name

    wandb_logger = wandb.init(dir=wandb_log_path, project=project_name,
                              name=train_logger.logger_name, config=args_train, tags=[tag]) if log_wandb else None

    trainer = TaskAdapter(
        model_name = args_train.model_name,
        num_labels = num_labels,
        dropout = args_train.task_dropout,
        n_hidden = args_train.task_n_hidden,
        rf = args_train.rf_task,
        bottleneck = args_train.bottleneck,
        bottleneck_dim = args_train.bottleneck_dim,
        bottleneck_dropout = args_train.bottleneck_dropout,
        encoder_state_dict = encoder_state_dict,
        wandb_logger = wandb_logger,
    )
    trainer.to(device)
    trainer_cp = trainer.fit(
        train_loader = train_loader,
        val_loader = val_loader,
        logger = train_logger,
        loss_fn = loss_fn,
        pred_fn = pred_fn,
        metrics = metrics,
        num_epochs = args_train.num_epochs,
        learning_rate = args_train.learning_rate,
        learning_rate_bottleneck = args_train.learning_rate_bottleneck,
        learning_rate_head = args_train.learning_rate_task_head,
        optimizer_warmup_steps = args_train.optimizer_warmup_steps,
        max_grad_norm = args_train.max_grad_norm,
        output_dir = args_train.output_dir,
        cooldown = args_train.cooldown,
        checkpoint_name = train_logger.logger_name + ".pt",
        seed = seed
    )

    trainer = TaskAdapter.load_checkpoint(trainer_cp)
    trainer.wandb_logger = wandb_logger
    trainer.to(device)
    test_result = trainer.evaluate(
        test_loader,
        loss_fn,
        pred_fn,
        metrics
    )
    test_str = {k: v for k, v in test_result.items()}
    if trainer.wandb_logger:
        wandb_test_str = {f"test task {k}": v for k, v in test_result.items()}
        trainer.wandb_logger.log(wandb_test_str)

    train_logger.writer.add_scalar("test/acc_task",test_str['acc'])
    train_logger.writer.add_scalar("test/bacc_task",test_str['bacc'])
    print('{:10}'.format("Final results test"))
    print(test_str)

    return trainer


def train_baseline_adv(
    device,
    train_loader,
    test_loader,
    val_loader,
    num_labels,
    num_labels_protected_list,
    protected_key_list,
    loss_fn,
    pred_fn,
    metrics,
    loss_fn_protected_list,
    pred_fn_protected_list,
    metrics_protected_list,
    train_logger,
    args_train,
    encoder_state_dict = None,
    task_head_state_dict = None,
    triplets = False,
    seed = None,
    log_wandb = 0,
):

    wandb_log_path = os.path.join(train_logger.log_dir, train_logger.logger_name)
    tag = str(train_logger.log_dir).split("_")[-1]

    project_name = f"{args_train.wandb_project}_{tag}"
    args_train.experiment_name = train_logger.logger_name

    wandb_logger = wandb.init(dir=wandb_log_path, project=project_name,
                              name=train_logger.logger_name, config=args_train, tags=[tag]) if log_wandb else None

    trainer = AdvModel(
        model_name = args_train.model_name,
        num_labels_task = num_labels,
        num_labels_protected = num_labels_protected_list,
        task_dropout = args_train.task_dropout,
        task_n_hidden = args_train.task_n_hidden,
        adv_dropout = args_train.adv_dropout,
        adv_n_hidden = args_train.adv_n_hidden,
        adv_count = args_train.adv_count,
        bottleneck = args_train.bottleneck,
        bottleneck_dim = args_train.bottleneck_dim,
        bottleneck_dropout = args_train.bottleneck_dropout,
        task_head_state_dict = task_head_state_dict,
        task_head_freeze = (task_head_state_dict is not None),
        encoder_state_dict = encoder_state_dict,
        wandb_logger = wandb_logger,
    )
    trainer.to(device)
    trainer_cp = trainer.fit(
        train_loader = train_loader,
        val_loader = val_loader,
        logger = train_logger,
        loss_fn = loss_fn,
        pred_fn = pred_fn,
        metrics = metrics,
        loss_fn_protected = loss_fn_protected_list,
        pred_fn_protected = pred_fn_protected_list,
        metrics_protected = metrics_protected_list,
        num_epochs = args_train.num_epochs,
        num_epochs_warmup = args_train.num_epochs_warmup,
        adv_lambda = args_train.adv_lambda,
        learning_rate = args_train.learning_rate,
        learning_rate_bottleneck = args_train.learning_rate_bottleneck,
        learning_rate_task_head = args_train.learning_rate_task_head,
        learning_rate_adv_head = args_train.learning_rate_adv_head,
        optimizer_warmup_steps = args_train.optimizer_warmup_steps,
        max_grad_norm = args_train.max_grad_norm,
        output_dir = args_train.output_dir,
        triplets = triplets,
        protected_key = protected_key_list,
        checkpoint_name = train_logger.logger_name + ".pt",
        seed = seed,
        training_method=args_train.training_method,
    )
    trainer = AdvModel.load_checkpoint(trainer_cp)

    trainer.wandb_logger = wandb_logger
    trainer.to(device)
    test_result = trainer.evaluate(
                    test_loader,
                    loss_fn,
                    pred_fn,
                    metrics
                )
    test_str = {k: v for k, v in test_result.items()}
    if trainer.wandb_logger:
        wandb_test_str = {f"test task {k}": v for k, v in test_result.items()}
        trainer.wandb_logger.log(wandb_test_str)

    train_logger.writer.add_scalar("test/acc_task", test_str['acc'])
    train_logger.writer.add_scalar("test/bacc_task", test_str['bacc'])
    print("Final results test ")
    print(test_str)

    return trainer

def train_adapter_adv(
    device,
    train_loader,
    test_loader,
    val_loader,
    num_labels,
    num_labels_protected_list,
    protected_key_list,
    loss_fn,
    pred_fn,
    metrics,
    loss_fn_protected_list,
    pred_fn_protected_list,
    metrics_protected_list,
    train_logger,
    args_train,
    encoder_state_dict = None,
    task_head_state_dict = None,
    triplets = False,
    seed = None,
    log_wandb = 0,
):

    wandb_log_path = os.path.join(train_logger.log_dir, train_logger.logger_name)
    tag = str(train_logger.log_dir).split("_")[-1]

    project_name = f"{args_train.wandb_project}_{tag}"
    args_train.experiment_name = train_logger.logger_name


    wandb_logger = wandb.init(dir=wandb_log_path, project=project_name,
                              name=train_logger.logger_name, config=args_train, tags=[tag]) if log_wandb else None

    trainer = AdvAdapter(
        model_name = args_train.model_name,
        num_labels_task = num_labels,
        num_labels_protected = num_labels_protected_list,
        task_dropout = args_train.task_dropout,
        task_n_hidden = args_train.task_n_hidden,
        adv_dropout = args_train.adv_dropout,
        adv_n_hidden = args_train.adv_n_hidden,
        rf = args_train.rf_task,
        adv_count = args_train.adv_count,
        bottleneck = args_train.bottleneck,
        bottleneck_dim = args_train.bottleneck_dim,
        bottleneck_dropout = args_train.bottleneck_dropout,
        task_head_state_dict = task_head_state_dict,
        task_head_freeze = (task_head_state_dict is not None),
        encoder_state_dict = encoder_state_dict,
        wandb_logger = wandb_logger,
    )

    trainer.to(device)
    trainer_cp = trainer.fit(
        train_loader = train_loader,
        val_loader = val_loader,
        logger = train_logger,
        loss_fn = loss_fn,
        pred_fn = pred_fn,
        metrics = metrics,
        loss_fn_protected = loss_fn_protected_list,
        pred_fn_protected = pred_fn_protected_list,
        metrics_protected = metrics_protected_list,
        num_epochs = args_train.num_epochs,
        num_epochs_warmup = args_train.num_epochs_warmup,
        adv_lambda = args_train.adv_lambda,
        learning_rate = args_train.learning_rate,
        learning_rate_bottleneck = args_train.learning_rate_bottleneck,
        learning_rate_task_head = args_train.learning_rate_task_head,
        learning_rate_adv_head = args_train.learning_rate_adv_head,
        optimizer_warmup_steps = args_train.optimizer_warmup_steps,
        max_grad_norm = args_train.max_grad_norm,
        output_dir = args_train.output_dir,
        triplets = triplets,
        protected_key = protected_key_list,
        checkpoint_name = train_logger.logger_name + ".pt",
        seed = seed,
        training_method=args_train.training_method,
    )

    trainer = AdvAdapter.load_checkpoint(trainer_cp)
    trainer.wandb_logger = wandb_logger
    trainer.to(device)
    test_result = trainer.evaluate(
        test_loader,
        loss_fn,
        pred_fn,
        metrics
    )

    test_str = {k: v for k, v in test_result.items()}
    if trainer.wandb_logger:
        wandb_test_str = {f"test task {k}": v for k, v in test_result.items()}
        trainer.wandb_logger.log(wandb_test_str)

    train_logger.writer.add_scalar("test/acc_task",test_str['acc'])
    train_logger.writer.add_scalar("test/bacc_task",test_str['bacc'])
    print("Final results test ")
    print(test_str)

    return trainer


def train_congater(
    device,
    train_loader,
    test_loader,
    val_loader,
    num_labels,
    num_labels_protected_list,
    protected_key_list,
    loss_fn,
    pred_fn,
    metrics,
    loss_fn_protected_list,
    pred_fn_protected_list,
    metrics_protected_list,
    train_logger,
    args_train,
    encoder_state_dict = None,
    task_head_state_dict = None,
    triplets = False,
    seed = None,
    log_wandb = 0,
):

    wandb_log_path = os.path.join(train_logger.log_dir, train_logger.logger_name)
    tag = str(train_logger.log_dir).split("_")[-1]

    project_name = f"{args_train.wandb_project}_{tag}"
    args_train.experiment_name = train_logger.logger_name

    wandb_logger = wandb.init(dir=wandb_log_path, project=project_name,
                              name=train_logger.logger_name, config=args_train, tags=[tag]) if log_wandb else None

    trainer = ConGaterModel(
        model_name=args_train.model_name,
        num_labels_task=num_labels,
        num_labels_protected=num_labels_protected_list,
        task_dropout=args_train.task_dropout,
        task_n_hidden=args_train.task_n_hidden,
        adv_dropout=args_train.adv_dropout,
        adv_n_hidden=args_train.adv_n_hidden,
        adv_count=args_train.adv_count,
        bottleneck=args_train.bottleneck,
        bottleneck_dim=args_train.bottleneck_dim,
        bottleneck_dropout=args_train.bottleneck_dropout,
        task_head_state_dict=task_head_state_dict,
        task_head_freeze=(task_head_state_dict is not None),
        encoder_state_dict=encoder_state_dict,
        congater_names=args_train.protected_key,
        congater_position=args_train.congater_position,
        congater_version=args_train.congater_version,
        num_gate_layers=args_train.num_gate_layers,
        gate_squeeze_ratio=args_train.gate_squeeze_ratio,
        default_trainable_parameters=args_train.default_trainable_parameters,
        custom_init=args_train.custom_init,
        no_scheduler= args_train.no_scheduler,
        wandb_logger=wandb_logger
    )

    trainer.set_trainable_parameters(args_train.default_trainable_parameters)

    print("Batch size:", args_train.batch_size)
    print("Name of the ConGater Submodules:", trainer.congater_names)
    print("Training Stages", trainer.training_stage)
    print("Training Method", args_train.training_method)
    print("Num and Squeeze in Each layer of Congater:", args_train.num_gate_layers, ",",  args_train.gate_squeeze_ratio,)
    print("Adversarial Lambda:", args_train.adv_lambda)
    print("Congater position", trainer.congater_position)
    print("ConGater Spec:", trainer.param_spec())
    print("Current Seed:", seed)
    print("Evaluation w:", args_train.evaluate_w)
    print("log_path:", wandb_log_path, "Logger Exist:", True if log_wandb else False)
    print("Training Device:", device)
    print("Congater Custom Initialization" if args_train.custom_init else "Congater Random Initialization")

    trainer.to(device)

    trainer_cp = trainer.fit(
        train_loader = train_loader,
        val_loader = val_loader,
        logger = train_logger,
        loss_fn = loss_fn,
        pred_fn = pred_fn,
        metrics = metrics,
        loss_fn_protected = loss_fn_protected_list,
        pred_fn_protected = pred_fn_protected_list,
        metrics_protected = metrics_protected_list,
        num_epochs = args_train.num_epochs,
        num_epochs_warmup = args_train.num_epochs_warmup,
        adv_lambda = args_train.adv_lambda,
        learning_rate = args_train.learning_rate,
        learning_rate_bottleneck = args_train.learning_rate_bottleneck,
        learning_rate_task_head = args_train.learning_rate_task_head,
        learning_rate_adv_head = args_train.learning_rate_adv_head,
        optimizer_warmup_steps = args_train.optimizer_warmup_steps,
        max_grad_norm = args_train.max_grad_norm,
        output_dir = args_train.output_dir,
        triplets = triplets,
        protected_key = protected_key_list,
        checkpoint_name = train_logger.logger_name + ".pt",
        seed = seed,
        training_method=args_train.training_method,
    )


    trainer = ConGaterModel.load_checkpoint(trainer_cp)
    trainer.wandb_logger = wandb_logger
    trainer.to(device)
    loss_fn_list = [loss_fn] + list(loss_fn_protected_list)
    pred_fn_list = [pred_fn] + list(pred_fn_protected_list)
    test_result= trainer.evaluate(
                    test_loader,
                    loss_fn_list,
                    pred_fn_list,
                    metrics,
                    trainer.training_stage
                )

    test_str = {k: v for k, v in test_result.items()}
    for stage in trainer.training_stage:
        train_logger.writer.add_scalar(f"test {stage} Acc", test_str[f'{stage}_acc'])
        train_logger.writer.add_scalar(f"test {stage} BAcc", test_str[f'{stage}_bacc'])
    print("Final results test ")
    print(test_str)

    for w in args_train.evaluate_w:
        performance = trainer.w_performance(
                    val_loader,
                    test_loader,
                    loss_fn_list,
                    pred_fn_list,
                    metrics,
                    w
                )
        print(f"model performance at omega={w}:")
        print(performance)

    return trainer


# (1) Finish ------training wrapper for # Baseline, BaselineADV, Adapter, AdapterADV and ConGater-----


def main():
# (2) Start --------Loading Config files and Arguments from command and setting up the runs ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", nargs="*", type=int, default=[0], help="select the gpu id")
    parser.add_argument("--model_type", type=str, default="baseline", help="values: baseline, baseline_adv, "
                                                                       "modular_baseline, modular_adv,"
                                                                       "adapter_baseline, adapter_prot, adapter_fusion,"
                                                                       "congater_baseline, congater, efficient")
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size of the Model")
    parser.add_argument("--seed", type=int, default=0, help="torch random seed")
    parser.add_argument("--lr", type=float, default=2e-4, help="torch random seed") #for GLUE 3e-3
    parser.add_argument("--random_seed", action="store_true", help="whether to use random seed or not")
    parser.add_argument("--ds", type=str, default="hatespeech", help="dataset")
    parser.add_argument("--num_runs", type=int, default=1, help="select number of times same experiment is running")
    parser.add_argument("--cpu", action="store_true", help="Run on cpu")
    parser.add_argument("--no_adv_attack", action="store_true", help="Set if you do not want to run probe attack after training")
    parser.add_argument("--cp_path", type=str, help="Overwrite pre-trained encoder weights")
    parser.add_argument("--cp_load_to_par", action="store_true", help="initialize checkpoint weights in parametrizations (doesent work for modular model)")
    parser.add_argument("--cp_load_task_head", action="store_true", help="load task head weights (doesent work for modular checkpoints)")
    parser.add_argument("--prot_key_idx", type=int, help="If protected key is type list: index of key to use, if none use all available attributes for taining")
    parser.add_argument("--debug", action="store_true", help="Whether to run on small subset for testing")
    parser.add_argument("--logger_suffix", type=str, help="Add addtional string to logger name")
    parser.add_argument("--model_name", type=str, default="mini", help="mini or base or roberta-base")
    parser.add_argument("--weighted_loss", action="store_true", default=False, help="apply weighted loss to the learning")
    parser.add_argument("--constant_task", action="store_true", help="set weights of task head to constant 1")
    parser.add_argument("--no_scheduler", action="store_true", help="Removes scheduler from the trainer")
    parser.add_argument("--congater_position", type=str, default="all", help="values: all and last")
    parser.add_argument("--default_trainable_parameters", type=str,
                        default="bert+task_head", help="changes the default parameters of bert during training")
    parser.add_argument("--custom_init", action="store_true", help="Congaters are custom Initialized")
    # TODO: Augmentation only applied on Baseline model
    parser.add_argument("--mixstyle", action="store_true", help="Mixstyle Augmentation")
    parser.add_argument("--training_method", type=str, default="par", help="type of training post of parallel")
    parser.add_argument("--num_gate_layers", type=int, default=2, help="number of layers in each congater")
    parser.add_argument("--gate_squeeze_ratio", type=int, default=4, help="scaling ration between layers of gating")
    parser.add_argument("--evaluate_w", type=list, default=[0, 0.2, 0.4, 0.6, 0.8, 1], help="Number of omega to run evaluation with")
    parser.add_argument("--congater_lambda", type=float, default=1, help="Congater Normalized Lambda value")

    parser.add_argument("--wandb_project", type=str, default="TEST", help="Project name, dataset will be added to project name automatically")
    parser.add_argument("--log_wandb", action="store_true", help="Whether to log on wandb or not")

    base_args, optional = parser.parse_known_args()
    for i in range(base_args.num_runs):
        with open("cfg.yml", "r") as f:
            cfg = yaml.safe_load(f)
        data_cfg = f"data_config_{base_args.ds}"

        if base_args.model_type == "congater":

            args_train = argparse.Namespace(**cfg["train_config"], **cfg[data_cfg],
                                            **cfg["model_config"], **cfg["congater_config"])

            args_train.default_trainable_parameters = base_args.default_trainable_parameters
            args_train.num_gate_layers = base_args.num_gate_layers
            args_train.custom_init = base_args.custom_init
            args_train.adv_lambda = base_args.congater_lambda
            args_train.gate_squeeze_ratio = base_args.gate_squeeze_ratio

            if type(args_train.protected_key) is list:
                args_train.evaluate_w = list(product(base_args.evaluate_w, base_args.evaluate_w))
            else:
                args_train.evaluate_w = base_args.evaluate_w

        else:

            args_train = argparse.Namespace(**cfg["train_config"], **cfg[data_cfg], **cfg["congater_config"],
                                            **cfg["model_config"])

        args_train.no_scheduler = base_args.no_scheduler
        args_train.congater_position = base_args.congater_position
        args_train.ds = base_args.ds
        args_train.learning_rate_bottleneck = base_args.lr
        args_train.log_dir = f"logs_{base_args.ds}"
        args_train.output_dir = f"checkpoints_{base_args.ds}"
        args_train.batch_size = base_args.batch_size
        args_train.constant_task = base_args.constant_task
        args_train.training_method = base_args.training_method
        args_train.weighted_loss_protected = base_args.weighted_loss
        args_train.wandb_project = base_args.wandb_project
        args_train.augment = base_args.mixstyle
        args_train.prot_key_idx = base_args.prot_key_idx

        # Default Config for different datasets tokenizer length
        if base_args.ds == "bios":
            args_train.tokenizer_max_length = 120
        elif base_args.ds in ["pan16", "hatespeech"]:
            args_train.tokenizer_max_length = 30

        args_attack = argparse.Namespace(**cfg["adv_attack"])
        set_optional_args(args_train, optional)
        if base_args.model_name == "base":
            args_train.model_name = "bert-base-uncased"
        elif base_args.model_name == "mini":
            args_train.model_name = "google/bert_uncased_L-4_H-256_A-4"
        elif base_args.model_name == "roberta-base":
            args_train.model_name = "roberta-base"
        else:
            print("Model name Not Recognized Please choose base , mini for bert models or roberta-base for roberta base")
            exit()

        args_train.model_type = base_args.model_type
        if base_args.debug:
            set_num_epochs_debug(args_train)
            set_num_epochs_debug(args_attack)
            set_dir_debug(args_train)

        if base_args.num_runs > 1:
            assert base_args.random_seed

        print("Run Number:", i+1)
        if base_args.random_seed:
            base_args.seed = torch.randint(0, int(1e9), (1, 1)).item()
        # print(f"args_train:\n{args_train}")

        torch.manual_seed(base_args.seed)
        print(f"torch.manual_seed({base_args.seed})")

        device = get_device(not base_args.cpu, base_args.gpu_id)
        print(f"Device: {device}")

        if base_args.cp_path is not None:
            encoder_cp = model_factory(
                cp_path = base_args.cp_path,
                remove_parametrizations = True,
                debiased = False # If loading checkpoint from modular model set debiased state
            )
            encoder_state_dict = encoder_cp.encoder.state_dict()
            if base_args.cp_load_task_head:
                task_head_state_dict = encoder_cp.task_head.state_dict()
            else:
                task_head_state_dict = None
        else:
            encoder_state_dict = None
            task_head_state_dict = None

        if base_args.ds in ["hatespeech", "bios", "pan16"]:

            train_loader, test_loader, val_loader, num_labels, num_labels_protected_list, protected_key_list, protected_class_weights_list = get_data(
                args_train = args_train,
                use_all_attr = (base_args.prot_key_idx is None),
                attr_idx_prot = base_args.prot_key_idx,
                compute_class_weights = args_train.weighted_loss_protected,
                device = device[0],
                triplets = args_train.triplets_loss,
                debug = base_args.debug
            )

        else:
            train_loader, test_loader, val_loader, num_labels, num_labels_protected_list, protected_key_list, protected_class_weights_list = get_glu_data(
                dataset = base_args.ds,
                args_train = args_train,
                debug = base_args.debug
            )

        print("Tokenizer Max Length:", args_train.tokenizer_max_length)
        loss_fn, pred_fn, metrics, loss_fn_protected_list, pred_fn_protected_list, metrics_protected_list = get_callables_wrapper(
            num_labels = num_labels,
            num_labels_protected = num_labels_protected_list,
            protected_class_weights = protected_class_weights_list
        )

        train_logger = get_logger(
            model_type = base_args.model_type,
            args_train = args_train,
            cp_path = (base_args.cp_path is not None),
            prot_key_idx = base_args.prot_key_idx,
            seed = base_args.seed,
            debug = base_args.debug,
            suffix = base_args.logger_suffix
        )
        # if base_args.modular:
 # (2) Finish --------Loading Config files and Arguments from command and setting up the runs ----------
        # (3) Start ------ Runing Training based on the selected models
        # To run adv on full model training
        if base_args.model_type == "baseline_adv":
            print("running train_baseline_adv")
            trainer = train_baseline_adv(device, train_loader,test_loader, val_loader, num_labels, num_labels_protected_list,
                                         protected_key_list, loss_fn, pred_fn, metrics, loss_fn_protected_list,
                                         pred_fn_protected_list, metrics_protected_list, train_logger, args_train,
                                         encoder_state_dict, task_head_state_dict,
                                         args_train.triplets_loss, base_args.seed, base_args.log_wandb
                                         )
        # Run task training on full model
        elif base_args.model_type == "baseline":
            print("running train_baseline_task")
            trainer = train_baseline_task(device, train_loader, test_loader, val_loader, num_labels, loss_fn, pred_fn, metrics,
                                           train_logger, args_train,  encoder_state_dict, base_args.seed, base_args.log_wandb)

        # run Adapter adv
        elif base_args.model_type == "adapter_adv":
            print("running train_adapter_adv")
            trainer = train_adapter_adv(device, train_loader,test_loader, val_loader, num_labels, num_labels_protected_list,
                                         protected_key_list, loss_fn, pred_fn, metrics, loss_fn_protected_list,
                                         pred_fn_protected_list, metrics_protected_list, train_logger, args_train,
                                         encoder_state_dict, task_head_state_dict, args_train.triplets_loss, base_args.seed,
                                         base_args.log_wandb)
        # run adapter baseline
        elif base_args.model_type == "adapter_baseline":
                print("running train_adapter_task")
                trainer = train_adapter_task(device, train_loader,test_loader, val_loader, num_labels, loss_fn, pred_fn, metrics,
                                              train_logger, args_train, encoder_state_dict, base_args.seed, base_args.log_wandb)
        # Run ConGater
        elif base_args.model_type == "congater":
            print("running train_congater ")
            trainer = train_congater(device, train_loader, test_loader, val_loader, num_labels, num_labels_protected_list,
                                     protected_key_list, loss_fn, pred_fn, metrics, loss_fn_protected_list,
                                     pred_fn_protected_list, metrics_protected_list, train_logger, args_train,
                                     encoder_state_dict, task_head_state_dict, args_train.triplets_loss, base_args.seed,
                                     base_args.log_wandb)
        # (3) Finish ------ Running Training based on the selected models

        wandb_logger = trainer.wandb_logger

        # If attacking at the end is required.
        # (4) Start ------ Attacking Trained models with Probes
        if not base_args.no_adv_attack:
            wandb_logger = None
            # ConGater Model has its own evaluation function because of the changes in omega value
            if base_args.model_type == "congater":
                wandb_logger = trainer.wandb_logger
                for w in args_train.evaluate_w:
                    if trainer.congater_names and (len(trainer.congater_names) > 1):
                        trainer.evaluate_w = dict(zip(trainer.congater_names, [value for value in w]))
                    elif trainer.congater_names and (len(trainer.congater_names) == 1):
                        trainer.evaluate_w = {trainer.congater_names[0]: w}
                    else:
                        trainer.evaluate_w = 0
                    print(f"Attack With omega: {trainer.evaluate_w}")
                    run_adv_attack(
                        base_args,
                        args_train,
                        args_attack,
                        trainer,
                        train_logger,
                        wandb_logger,
                    )

            else:
                wandb_logger = trainer.wandb_logger

                run_adv_attack(
                    base_args,
                    args_train,
                    args_attack,
                    trainer,
                    train_logger,
                    wandb_logger
                )
        # (4) Finish ------ Attacking Trained models with Probes
        if wandb_logger:
            wandb_logger.finish()

        del trainer


if __name__ == "__main__":

    main()
