# %%
import torch
from torch import nn

import distutils.version
from torch.utils.tensorboard import SummaryWriter
from model_congater import GateLayer
from model_congater import ConGaterModel
from adapterbaseline import AdapterModel
from model_interpolate import ModelInterpolate
from modeling import MDSTransformer
from main import get_query_encoder, get_loss_module, setup, get_tokenizer, get_dataset, train, get_model, main
from options import run_parse_args
import utils
from transformers import AutoModel
from torch.utils.data import DataLoader
from dataset import MYMARCO_Dataset
from optimizers import get_optimizers
from fair_retrieval.metrics_FaiRR import FaiRRMetric, FaiRRMetricHelper
#from utils import load_model

if __name__ == "__main__":
    args = run_parse_args()
    config = setup(args)
    args = utils.dict2obj(config)


    bert = AutoModel.from_pretrained("google/bert_uncased_L-4_H-256_A-4")

    # %%
    cong = GateLayer(embed_size=765, squeeze_ratio=2)
    cong


    # %%

    doc_emb_dim=768
 
    query_encoder = get_query_encoder(args.query_encoder_from, args.query_encoder_config)

    # %%
    query_encoder

    # %%
    loss_module = get_loss_module(args.loss_type, args)
    aux_loss_module = None

# irmodel = MDSTransformer(custom_encoder=query_encoder,
#                               d_model=args.d_model,
#                               num_heads=args.num_heads,
#                               num_decoder_layers=args.num_layers,
#                               dim_feedforward=args.dim_feedforward,
#                               dropout=args.dropout,
#                               activation=args.activation,
#                               normalization=args.normalization_layer,
#                               doc_emb_dim=doc_emb_dim,
#                               scoring_mode=args.scoring_mode,
#                               query_emb_aggregation=args.query_aggregation,
#                               loss_module=loss_module,
#                               aux_loss_module=aux_loss_module,
#                               aux_loss_coeff=args.aux_loss_coeff,
#                               selfatten_mode=args.selfatten_mode,
#                               no_decoder=args.no_decoder,
#                               no_dec_crossatten=args.no_dec_crossatten,
#                               bias_regul_coeff=args.bias_regul_coeff,
#                               bias_regul_cutoff=args.bias_regul_cutoff)
# print(irmodel.hidden_size)


    # %%
args.n_gpu = len(args.cuda_ids)
if torch.cuda.is_available():
    if args.n_gpu > 1:
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cuda:%d" % args.cuda_ids[0])
else:
    args.device = torch.device("cpu")
tokenizer = get_tokenizer(args)
eval_dataset = get_dataset(args, 'dev', tokenizer)  # CHANGED here from eval_mode
collate_fn = eval_dataset.get_collate_func(n_gpu=args.n_gpu)

args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
# Note that DistributedSampler samples randomly
eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                 num_workers=args.data_num_workers, collate_fn=collate_fn)
train_dataset = MYMARCO_Dataset('train', args.embedding_memmap_dir, args.tokenized_path, args.train_candidates_path,
                                qrels_path=args.qrels_path, tokenizer=tokenizer,
                                max_query_length=args.max_query_length, num_candidates=args.num_candidates,
                                limit_size=args.train_limit_size,
                                load_collection_to_memory=args.load_collection_to_memory,
                                emb_collection=eval_dataloader.dataset.emb_collection,
                                include_zero_labels=args.include_zero_labels,
                                relevance_labels_mapping=args.relevance_labels_mapping,
                                collection_neutrality_path=args.collection_neutrality_path,
                                query_ids_path=args.train_query_ids)
collate_fn = train_dataset.get_collate_func(num_random_neg=args.num_random_neg, n_gpu=args.n_gpu)

# NOTE RepBERT: Must be sequential! Pos, Neg, Pos, Neg, ...
# This is because a (query, pos. doc, neg. doc) triplet is split in 2 consecutive samples: (qID, posID) and (qID, negID)
# If random sampling had been chosen, then these 2 samples would have ended up in different batches
# train_sampler = SequentialSampler(train_dataset)

train_dataloader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=32, num_workers=args.data_num_workers,
                                  collate_fn=collate_fn)


# irmodel = get_model(args, eval_dataset.emb_collection.embedding_vectors.shape[1])

# for step, (model_inp, _, _) in enumerate(train_dataloader):
#     print(step)
#     print([v.shape for k,v in model_inp.items()])
#     break
# model_inp = {k: v.to(args.device) for k, v in model_inp.items()}
# irmodel.to(args.device)
# output = irmodel(**model_inp)
# output


fairrmetric = None
if args.collection_neutrality_path is not None:
    _fairrmetrichelper = FaiRRMetricHelper()
    _background_doc_set = _fairrmetrichelper.read_documentset_from_retrievalresults(args.background_set_runfile_path)
    fairrmetric = FaiRRMetric(args.collection_neutrality_path, _background_doc_set)


# %%
#TODO: check out forward function and outputs
#output = irmodel(**model_inp)
#nonencoder_optimizer, encoder_optimizer = get_optimizers(args, irmodel)

# %%

    ##new congater class
congmodel = ModelInterpolate(custom_encoder=query_encoder,
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
                              bias_regul_cutoff=args.bias_regul_cutoff)
print([n for n, _ in congmodel.named_children()])

utils.load_adapter(congmodel,"/mnt/c/users/cornelia/documents/AI/MasterThesis/IRDebias/coderdebiasinf/results/adapter/tests_2023-09-03_16-39-31_p9X/checkpoints/gender/model_78590.pth")


#congmodel.to(args.device)
# output = congmodel(**model_inp, w = {"gender": 0.2}, bias_regul_coeff = 0)
# output

# nonencoder_optimizer, encoder_optimizer = get_optimizers(args, congmodel)
# nonencoder_optimizer



