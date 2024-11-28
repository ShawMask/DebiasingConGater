import logging

import options

logging.basicConfig(format='%(asctime)s | %(name)-8s - %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()
logger.info("Loading packages ...")
import os
import sys
import random
import time
import json
from datetime import datetime
import string
from collections import OrderedDict

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import torch
import distutils.version
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler
# from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers import BertConfig, AutoTokenizer, AutoModel
import optuna
from model_congater import ConGaterModel
from adapterbaseline import AdapterModel
from model_interpolate import AdapterInterpolateModel
import math

# Package modules
from options import *
from modeling import RepBERT_Train, MDSTransformer, get_loss_module
from dataset import MYMARCO_Dataset, MSMARCODataset
from dataset import lookup_times, sample_fetching_times, collation_times, retrieve_candidates_times, prep_docids_times
import utils
from utils import save_inference
from fair_retrieval.metrics_FaiRR import FaiRRMetric, FaiRRMetricHelper

#val_times = utils.Timer()  # stores measured validation times


def train(args, model, val_dataloader, tokenizer=None, fairrmetric=None, trial=None):
    """
    Prepare training dataset, train the model and handle results.
    fairrmetric is an optional object for evaluating fairness/neutrality of ranked documents.
    trial is an optional Optuna hyperparameter optimization object
    """
    tb_writer = SummaryWriter(args.tensorboard_dir)

    train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    logger.info("Preparing {} dataset ...".format('train'))
    start_time = time.time()
    train_dataset = MYMARCO_Dataset('train', args.embedding_memmap_dir, args.tokenized_path, args.train_candidates_path,
                                    qrels_path=args.qrels_path, tokenizer=tokenizer,
                                    max_query_length=args.max_query_length, num_candidates=args.num_candidates,
                                    limit_size=args.train_limit_size,
                                    load_collection_to_memory=args.load_collection_to_memory,
                                    emb_collection=val_dataloader.dataset.emb_collection,
                                    include_zero_labels=args.include_zero_labels,
                                    relevance_labels_mapping=args.relevance_labels_mapping,
                                    collection_neutrality_path=args.collection_neutrality_path,
                                    query_ids_path=args.train_query_ids)
    collate_fn = train_dataset.get_collate_func(num_random_neg=args.num_random_neg, n_gpu=args.n_gpu)
    logger.info("'train' data loaded in {:.3f} sec".format(time.time() - start_time))

    utils.write_list(os.path.join(args.output_dir, "train_IDs.txt"), train_dataset.qids)
    utils.write_list(os.path.join(args.output_dir, "val_IDs.txt"), val_dataloader.dataset.qids)

    # NOTE RepBERT: Must be sequential! Pos, Neg, Pos, Neg, ...
    # This is because a (query, pos. doc, neg. doc) triplet is split in 2 consecutive samples: (qID, posID) and (qID, negID)
    # If random sampling had been chosen, then these 2 samples would have ended up in different batches
    # train_sampler = SequentialSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=train_batch_size, num_workers=args.data_num_workers,
                                  collate_fn=collate_fn)

    epoch_steps = (len(train_dataloader) // args.grad_accum_steps)  # num. actual steps (param. updates) per epoch
    total_training_steps = epoch_steps * args.num_epochs * len(model.training_stage)
    start_step = 0  # which step training started from

    # Load model and possibly optimizer/scheduler state
    optim_state, sched_state = None, None
    if args.load_model_path:  # model is already on its intended device, which we pass as an argument
        model, start_step, optim_state, sched_state = utils.load_model(model, args.load_model_path, args.device, args.resume)
        if args.load_adapter_path:  # get additional adapter parameters from file
            model, _ = utils.load_adapter(model, args.load_adapter_path, args.device)

    # Prepare optimizer and schedule
    sstr = ['non-encoder', 'encoder']
    model._init_optimizer_and_schedule(args, logger, start_step, optim_state, total_training_steps, sched_state)

    global_step = start_step  # counts how many times the weights have been updated, i.e. num. batches // gradient acc. steps
    start_epoch = global_step // epoch_steps
    if start_step >= total_training_steps:
        logger.error("The loaded model has been already trained for {} steps ({} epochs), "
                     "while specified `num_epochs` is {} (total steps {})".format(start_epoch, start_step,
                                                                                  args.num_epochs,
                                                                                  total_training_steps))
        sys.exit(1)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=args.cuda_ids)

    # Initialize performance tracking and evaluate model before training
    best_values = []  # sorted list of length args.num_keep_best, containing the top args.num_keep_best performance metrics
    best_steps = []  # list containing the global step number corresponding to best_values
    running_metrics = []  # (for validation) list of lists: for every evaluation, stores metrics like loss, MRR, MAP, ...
    train_loss = 0  # this is the training loss accumulated from the beginning of training
    logging_loss = 0  # this is synchronized with `train_loss` every args.logging_steps
    val_metrics = dict()
    #init all the model training metrics and steps:
    model.init_training_par(global_step, best_values, best_steps, running_metrics, val_metrics, train_loss, logging_loss, best_metrics = dict())

    # Train
    logger.info("\n\n***** START TRAINING *****\n\n")
    logger.info("Number of epochs: %d", args.num_epochs)
    logger.info("Number of training examples: %d", len(train_dataset))
    logger.info("Number of validation examples: {}".format(len(val_dataloader.dataset)))
    logger.info("Batch size per GPU: %d", args.per_gpu_train_batch_size)
    logger.info("Total train batch size (w. parallel, distributed & accumulation): %d",
                train_batch_size * args.grad_accum_steps)
    logger.info("Gradient Accumulation steps: %d", args.grad_accum_steps)
    logger.info("Total optimization steps: %d", total_training_steps)
    for i, s in enumerate(model.scheduler.get_last_lr()):
        logger.debug('Learning rate ({}): {}'.format(sstr[i], s))

    if args.congater_training_method == 'post' or args.congater_training_method is None:
        for label_idx, stage in enumerate(model.training_stage):

            logger.info("\n\n***** Initial evaluation on dev set *****".format(model.global_step))
            #Call validation function
            model.val_metrics = model.validate(args, val_dataloader, tb_writer, [stage], model.bias_regul_coeff, logger, fairrmetric=fairrmetric)

            model.best_metrics = model.val_metrics.copy()  # dict of all monitored metrics at the step with the best args.key_metric
            model.metrics_names, model.metrics_values = zip(*model.val_metrics.items())
            model.running_metrics.append(list(model.metrics_values))

            model.zero_grad()
            model.train()
            epoch_iterator = trange(start_epoch, int(args.num_epochs+args.num_epochs_warmup), desc="Epochs")

            batch_times = utils.Timer()  # average time for the model to train (forward + backward pass) on a single batch of queries
            for epoch_idx in epoch_iterator:
                epoch_start_time = time.time()

                if epoch_idx < args.num_epochs_warmup:
                    bias_regul_coeff = 0.
                else:
                    bias_regul_coeff = model.bias_regul_coeff

                #####################################################
                #Call step function here
                #####################################################

                step = model._step(args,train_dataloader, val_dataloader, logger, tb_writer, [stage], bias_regul_coeff, batch_times, total_training_steps, fairrmetric, trial)


                epoch_runtime = time.time() - epoch_start_time
                logger.info("Epoch runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(epoch_runtime)))

            # Export record metrics to a file accumulating records from all experiments
            utils.register_record(args.records_file, args.initial_timestamp, args.experiment_name,
                                model.best_metrics, model.val_metrics, comment=args.comment, stage = stage)
    else: #other training_methods can be applied but are not implemented at this point
        raise NotImplementedError
    
    # Export evolution of metrics over epochs, stage is added to logging
    header = model.metrics_names + ('Stage',)
    metrics_filepath = os.path.join(args.output_dir, "metrics_" + args.experiment_name + ".xls")
    if len(model.training_stage) > 1:
        task_list = [model.training_stage[0]] + np.repeat(model.training_stage,int(len(model.running_metrics)/len(model.training_stage))).tolist()
    else:
        task_list = np.repeat(model.training_stage,int(len(model.running_metrics))).tolist()
    print(task_list)
    print(len(model.running_metrics))
    book = utils.export_performance_metrics(metrics_filepath, [x+[task_list[i]] for i, x in enumerate(model.running_metrics)], header, sheet_name="metrics")

    avg_batch_time = batch_times.total_time / (epoch_idx * epoch_steps + step + 1)
    logger.info("Average time to train on 1 batch ({} samples): {:.6f} sec"
                " ({:.6f}s per sample)".format(train_batch_size, avg_batch_time, avg_batch_time / train_batch_size))
    logger.info("Average time to train on 1 batch ({} samples): {:.6f} sec".format(train_batch_size, batch_times.get_average()))
    logger.info('Best {} was {}. Other metrics: {}'.format(args.key_metric, model.best_values[0], model.best_metrics))

    logger.debug("Average lookup time: {} s".format(lookup_times.get_average()))
    logger.debug("Average retr. candidates time: {} s".format(retrieve_candidates_times.get_average()))
    logger.debug("Average prep. docids time: {} s".format(prep_docids_times.get_average()))
    logger.debug("Average sample fetching time: {} s".format(sample_fetching_times.get_average()))
    logger.debug("Average collation time: {} s".format(collation_times.get_average()))

    logger.info("Current memory usage: {} MB".format(np.round(utils.get_current_memory_usage())))
    logger.info("Max memory usage: {} MB".format(int(np.ceil(utils.get_max_memory_usage()))))

    return model.best_metrics


def main(config, trial=None):  # trial is an Optuna hyperparameter optimization object
    args = utils.dict2obj(config)  # Convert config dict to args object

    if args.debug:
        logger.setLevel('DEBUG')
    # Add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(args.output_dir, 'output.log'))
    logger.addHandler(file_handler)

    logger.info('Running:\n{}\n'.format(' '.join(sys.argv)))  # command used to run

    # Setup CUDA, GPU
    args.n_gpu = len(args.cuda_ids)
    if torch.cuda.is_available():
        if args.n_gpu > 1:
            args.device = torch.device("cuda")
        else:
            args.device = torch.device("cuda:%d" % args.cuda_ids[0])
    else:
        args.device = torch.device("cpu")

    # Log current hardware setup
    logger.info("Device: %s, n_gpu: %s", args.device, args.n_gpu)
    if args.device.type == 'cuda':
        logger.info("Device: {}".format(torch.cuda.get_device_name(0)))
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
        logger.info("Total memory: {} MB".format(np.ceil(total_mem)))


    # Set seed
    utils.set_seed(args)

    # Get tokenizer
    tokenizer = get_tokenizer(args)

    # Load evaluation set and initialize evaluation dataloader
    if args.task == 'train':
        eval_mode = 'dev'  # 'eval' here is the name of the MSMARCO test set, 'dev' is the validation set
    else:
        eval_mode = args.task

    logger.info("Preparing {} dataset ...".format(eval_mode))
    start_time = time.time()
    eval_dataset = get_dataset(args, eval_mode, tokenizer)  # CHANGED here from eval_mode
    collate_fn = eval_dataset.get_collate_func(n_gpu=args.n_gpu)
    logger.info("'{}' data loaded in {:.3f} sec".format(eval_mode, time.time() - start_time))

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                 num_workers=args.data_num_workers, collate_fn=collate_fn)

    logger.info("Number of {} samples: {}".format(eval_mode, len(eval_dataset)))
    logger.info("Batch size: %d", args.eval_batch_size)

    # initialize fairness
    fairrmetric = None
    if args.collection_neutrality_path is not None:
        logger.info("Loading FaiRRMetric ...")
        _fairrmetrichelper = FaiRRMetricHelper()
        _background_doc_set = _fairrmetrichelper.read_documentset_from_retrievalresults(args.background_set_runfile_path)
        fairrmetric = FaiRRMetric(args.collection_neutrality_path, _background_doc_set)

    # Initialize model. This is done after loading the data, to know the doc. embeddings dimension
    logger.info("Initializing model ...")
    if args.model_type == 'repbert':
        # keep configuration setup like RepBERT (for backward compatibility).
        # The model is a common/shared BERT query-document encoder, without interactions between query and document token representations
        if args.load_model_path is None:
            args.load_model_path = "bert-base-uncased"
        # Works with either directory path containing HF config file, or JSON HF config file,  or pre-defined model string
        config_obj = BertConfig.from_pretrained(args.load_model_path)
        model = RepBERT_Train.from_pretrained(args.load_model_path, config=config_obj)
    else:  # new configuration setup for MultiDocumentScoringTransformer models
        model = get_model(args, eval_dataset.emb_collection.embedding_vectors.shape[1])

    logger.debug("Model:\n{}".format(model))
    logger.info("Total number of model parameters: {}".format(utils.count_parameters(model)))
    logger.info("Total trainable parameters: {}".format(utils.count_parameters(model, trainable=True)))
    logger.info("Number of encoder parameters: {}".format(utils.count_parameters(model.encoder)))
    logger.info("Trainable encoder parameters: {}".format(utils.count_parameters(model.encoder, trainable=True)))

    model.to(args.device)  # will also print model architecture, besides moving to GPU

    if args.task == "train":
        return train(args, model, eval_dataloader, tokenizer, fairrmetric=fairrmetric, trial=trial)
    else:
        # Just evaluate trained model on some dataset (needs ~27GB for MS MARCO dev set)

        # only composite (non-repbert) models need to be loaded; repbert is already loaded at this point
        if args.load_model_path and (args.model_type != 'repbert'):
            model, global_step, _, _ = utils.load_model(model, args.load_model_path, device=args.device)
            if args.load_adapter_path:  # get additional adapter parameters from file, can be used to load trained adapter models on other model
                model = utils.load_adapter(model, args.load_adapter_path, args.device)

        logger.info("Will evaluate model on candidates in: {}".format(args.eval_candidates_path))

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=args.cuda_ids)

        model.eval()
        # Run evaluation
        eval_start_time = time.time()

        #no inspect mode available for congater
        #evaluate for the w values given at inference time
        if args.evaluate_w is not None:
            if config['congater_names']:
                for w in args.evaluate_w:
                    if type(w) is int or type(w) is float:
                        w = [w]
                    w = dict(zip(model.congater_names, [value for value in w])) if model.congater_names else {"task": 0}
                    eval_metrics, ranked_df = model.eval_step(args, eval_dataloader, args.bias_regul_coeff, w, logger, fairrmetric=fairrmetric)
                    save_inference(args, eval_metrics, ranked_df, w = w[model.congater_names[0]])
            if config['adapter_names']:
                for w in args.evaluate_w:
                    if type(w) is int or type(w) is float:
                        w = [w]
                    w = dict(zip([model.adapter_names], [value for value in w]))
                    eval_metrics, ranked_df = model.eval_step(args, eval_dataloader, args.bias_regul_coeff, w, logger, fairrmetric=fairrmetric)
                    save_inference(args, eval_metrics, ranked_df, w = w[model.adapter_names])
        #assuming no evaluate_w values are set
        else:
            eval_metrics, ranked_df = model.eval_step(args, eval_dataloader, args.bias_regul_coeff, logger, fairrmetric=fairrmetric)
            save_inference(args, eval_metrics, ranked_df, w = None)
        eval_runtime = time.time() - eval_start_time
        logger.info("Evaluation runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(eval_runtime)))
        print()

        return eval_metrics


def setup(args):
    """Prepare training session: read configuration from file (takes precedence), create directories.
    Input:
        args: arguments object from argparse
    Returns:
        config: configuration dictionary
    """

    config = utils.load_config(args)  # configuration dictionary
    config = options.check_args(config)  # check validity of settings and make necessary conversions

    # Create output directory and subdirectories
    initial_timestamp = datetime.now()
    output_dir = config['output_dir']
    if not os.path.isdir(output_dir):
        raise IOError(
            "Root directory '{}', where the directory of the experiment will be created, must exist".format(output_dir))

    output_dir = os.path.join(output_dir, config['experiment_name'])

    formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    if (not config['no_timestamp']) or (len(config['experiment_name']) == 0):
        rand_suffix = "".join(random.choices(string.ascii_letters + string.digits, k=3))
        output_dir += "_" + formatted_timestamp + "_" + rand_suffix
    config['output_dir'] = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)
    logger.info("Stored configuration file in '{}'".format(output_dir))

    # Create subdirectories and store additional configuration info
    info_dict = {'initial_timestamp': formatted_timestamp,
                 'save_dir': os.path.join(output_dir, 'checkpoints'),
                 'pred_dir': os.path.join(output_dir, 'predictions'),
                 'tensorboard_dir': os.path.join(output_dir, 'tb_summaries')}
    utils.create_dirs([info_dict['save_dir'], info_dict['pred_dir'], info_dict['tensorboard_dir']])
    #if congater names are set create subfolders in checkpoints and predictions
    if config['congater_names']:
        training_stages = ["task"] + config['congater_names']
        utils.create_dirs([os.path.join(info_dict['save_dir'], f) for f in training_stages])
        utils.create_dirs([os.path.join(info_dict['pred_dir'], f) for f in training_stages])
    if config['adapter_names']:
        utils.create_dirs([os.path.join(info_dict['save_dir'], config['adapter_names'])])
        utils.create_dirs([os.path.join(info_dict['pred_dir'], config['adapter_names'])])
    with open(os.path.join(output_dir, 'info.txt'), 'w') as fp:
        json.dump(info_dict, fp)
    config.update(info_dict)

    return config


def get_dataset(args, eval_mode, tokenizer):
    """Initialize and return evaluation dataset object based on args"""

    if args.model_type == 'repbert':
        return MSMARCODataset(eval_mode, args.msmarco_dir, args.collection_memmap_dir, args.tokenized_path,
                              args.max_query_length, args.max_doc_length, limit_size=args.eval_limit_size)
    else:
        return MYMARCO_Dataset(eval_mode, args.embedding_memmap_dir, args.eval_query_tokens_path,
                               args.eval_candidates_path, qrels_path=args.qrels_path, tokenizer=tokenizer,
                               max_query_length=args.max_query_length,
                               num_candidates=None,  # Always use ALL candidates for evaluation
                               limit_size=args.eval_limit_size,
                               load_collection_to_memory=args.load_collection_to_memory,
                               inject_ground_truth=args.inject_ground_truth,
                               include_zero_labels=args.include_zero_labels,
                               relevance_labels_mapping=args.relevance_labels_mapping,
                               query_ids_path=args.eval_query_ids)


def get_query_encoder(query_encoder_from, query_encoder_config):
    """Initialize and return query encoder model object based on args"""

    if os.path.exists(query_encoder_from):
        logger.info("Will load pre-trained query encoder from: {}".format(query_encoder_from))
    else:
        logger.warning("Will initialize standard HuggingFace '{}' as a query encoder!".format(query_encoder_from))
    start_time = time.time()
    encoder = AutoModel.from_pretrained(query_encoder_from, config=query_encoder_config)
    logger.info("Query encoder loaded in {} s".format(time.time() - start_time))
    return encoder


def get_model(args, doc_emb_dim=None):
    """Initialize and return end-to-end model object based on args"""

    query_encoder = get_query_encoder(args.query_encoder_from, args.query_encoder_config)
    loss_module = get_loss_module(args.loss_type, args)
    aux_loss_module = None
    if args.aux_loss_type is not None:
        aux_loss_module = get_loss_module(args.aux_loss_type, args)  # instantiate auxiliary loss module

    if args.model_type == 'congater':
        return ConGaterModel(custom_encoder=query_encoder,
                              d_model=args.d_model,
                              num_heads=args.num_heads,
                              num_decoder_layers=args.num_layers,
                              dim_feedforward=args.dim_feedforward,
                              dropout=args.dropout,
                              activation=args.activation,
                              normalization=args.normalization_layer,
                              doc_emb_dim=doc_emb_dim,
                              scoring_mode=args.scoring_mode,
                              query_emb_aggregation=args.query_aggregation,
                              loss_module=loss_module,
                              aux_loss_module=aux_loss_module,
                              aux_loss_coeff=args.aux_loss_coeff,
                              selfatten_mode=args.selfatten_mode,
                              no_decoder=args.no_decoder,
                              no_dec_crossatten=args.no_dec_crossatten,
                              bias_regul_coeff=args.bias_regul_coeff,
                              bias_regul_cutoff=args.bias_regul_cutoff,
                              congater_names=args.congater_names,
                              congater_position=args.congater_position,
                              congater_version=args.congater_version,
                              num_gate_layers=args.num_gate_layers,
                              gate_squeeze_ratio=args.gate_squeeze_ratio,
                              default_trainable_parameters=args.default_trainable_parameters)
    
    elif args.model_type == 'adapter':
        return AdapterModel(custom_encoder=query_encoder,
                              d_model=args.d_model,
                              num_heads=args.num_heads,
                              num_decoder_layers=args.num_layers,
                              dim_feedforward=args.dim_feedforward,
                              dropout=args.dropout,
                              activation=args.activation,
                              normalization=args.normalization_layer,
                              doc_emb_dim=doc_emb_dim,
                              scoring_mode=args.scoring_mode,
                              query_emb_aggregation=args.query_aggregation,
                              loss_module=loss_module,
                              aux_loss_module=aux_loss_module,
                              aux_loss_coeff=args.aux_loss_coeff,
                              selfatten_mode=args.selfatten_mode,
                              no_decoder=args.no_decoder,
                              no_dec_crossatten=args.no_dec_crossatten,
                              bias_regul_coeff=args.bias_regul_coeff,
                              bias_regul_cutoff=args.bias_regul_cutoff,
                              squeeze_ratio=args.squeeze_ratio,
                              adapter_names=args.adapter_names)
    if args.model_type == 'adapterinterpolate':
                return AdapterInterpolateModel(custom_encoder=query_encoder,
                              d_model=args.d_model,
                              num_heads=args.num_heads,
                              num_decoder_layers=args.num_layers,
                              dim_feedforward=args.dim_feedforward,
                              dropout=args.dropout,
                              activation=args.activation,
                              normalization=args.normalization_layer,
                              doc_emb_dim=doc_emb_dim,
                              scoring_mode=args.scoring_mode,
                              query_emb_aggregation=args.query_aggregation,
                              loss_module=loss_module,
                              aux_loss_module=aux_loss_module,
                              aux_loss_coeff=args.aux_loss_coeff,
                              selfatten_mode=args.selfatten_mode,
                              no_decoder=args.no_decoder,
                              no_dec_crossatten=args.no_dec_crossatten,
                              bias_regul_coeff=args.bias_regul_coeff,
                              bias_regul_cutoff=args.bias_regul_cutoff,
                              squeeze_ratio=args.squeeze_ratio,
                              adapter_names=args.adapter_names,
                              interpolate_position = args.interpolate_position)
    else:
        raise NotImplementedError('Unknown model type')


def get_tokenizer(args):
    """Initialize and return tokenizer object based on args"""

    if args.tokenizer_from is None:  # use same config as specified for the query encoder model
        return AutoTokenizer.from_pretrained(args.query_encoder_from, config=args.query_encoder_config)
    else:
        return AutoTokenizer.from_pretrained(args.tokenizer_from)


if __name__ == "__main__":
    total_start_time = time.time()
    args = run_parse_args()
    config = setup(args)  # Setup experiment session
    main(config)
    logger.info("All done!")
    total_runtime = time.time() - total_start_time
    logger.info("Total runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))
