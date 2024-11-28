import os
import math
from tqdm import trange, tqdm
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.functional import triplet_margin_loss
from collections import OrderedDict
import numpy as np

from typing import Union, Callable, Dict, Optional
import bisect
from options import *
from itertools import product

from src.models.model_heads import ClfHead
from modeling import MDSTransformer
import optuna
import time
from utils import Timer, get_current_memory_usage, get_current_memory_usage2, get_max_memory_usage, calculate_metrics, get_relevances, rank_docs, save_inference, ascii_bar_plot, register_record, reverse_bisect_right, save_model, readable_time, remove_oldest_checkpoint
import logging
from torch.utils.tensorboard import SummaryWriter
from dataset import lookup_times, sample_fetching_times, collation_times, retrieve_candidates_times, prep_docids_times
from optimizers import get_optimizers, MultiOptimizer, get_schedulers, MultiScheduler
import sys
import pandas as pd

val_times = Timer()  # stores measured validation times

STEP_THRESHOLD = 0  #4000 # Used for fairness regularization; checkpoints corresponding to best performance before STEP_THRESHOLD steps will be ignored

#First version of Tsigmoid function (version 1)
class Tsigmoid(nn.Module):
    def __init__(self):
        super(Tsigmoid, self).__init__()
    def forward(self, x, w):
        out = 1 - torch.log2(torch.tensor(w+1)) / (1 + torch.exp(x))  # 0<w<1
        return out

#Other version of Tsigmoid function (version 2)
class TTsigmoid(nn.Module):
    def __init__(self):
        super(TTsigmoid, self).__init__()

    def forward(self, x: torch.Tensor, w: float = 1):
        out = torch.log2(torch.tensor(w + 1)) / (1 + torch.exp(-x)) # 0<w<1
        return out


# Congater linear layer, can also be multiple layers
class GateLayer(nn.Module):
    def __init__(self, embed_size, num_layers=2, squeeze_ratio=4, version=1):
        super(GateLayer, self).__init__()
        #depth of linear layers
        self.num_layers = num_layers
        #Tsigmoid version
        self.gate_version = version
        #list of embeding sizes that differ with squeeze ration
        embed_size = [embed_size] + [embed_size//((i+1)*squeeze_ratio) for i in range(num_layers-1)]

        if num_layers == 0:
            self.linear = nn.Parameter(torch.rand(embed_size, requires_grad=True))
        else:
            self.linear = nn.ModuleList()
            for i in range(num_layers):
                if i < (num_layers - 1):
                    self.linear.append(nn.Sequential(nn.Linear(embed_size[i], embed_size[i+1]), nn.Tanh()))
                else:
                    self.linear.append(nn.Linear(embed_size[-1], embed_size[0]))

        if version == 1:
            self.activation = Tsigmoid()
        elif version == 2:
            self.activation = TTsigmoid()
        else:
            print(" Warning Activation Function is not within the known versions")
            exit()

    #Input is the transformer output embedding and w the hyperparameter
    def forward(self, embed, w):
        if self.num_layers == 0:
            embed = self.linear
        else:
            for i in range(self.num_layers):
                embed = self.linear[i](embed)
        if self.gate_version == 1:
            return self.activation(embed, w)
        elif self.gate_version == 2:
            return embed*self.activation(embed, w)
        


class ConGaterModel(MDSTransformer):

    def __init__(
        self,
        bottleneck: bool = False,
        bottleneck_dim: Optional[int] = None,
        bottleneck_dropout: Optional[float] = None,
        congater_names: Optional[str] = "gender",
        congater_position: str = "all",
        congater_version: int = 1,
        num_gate_layers: int = 1,
        gate_squeeze_ratio: int = 4,
        default_trainable_parameters: str = "all",
        custom_init: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)


        self.has_bottleneck = bottleneck
        self.bottleneck_dim = bottleneck_dim
        self.bottleneck_dropout = bottleneck_dropout
        self.congater_position = congater_position
        self.congater_version = congater_version
        self.num_gate_layers = num_gate_layers
        self.gate_squeeze_ratio = gate_squeeze_ratio
        self.global_step = 0
        self.evaluate_w = {"task": 0}
        self.custom_init = custom_init
        if congater_names:
            self.congater_names = congater_names if type(congater_names) is list else [congater_names]
            self.evaluate_w = list(product(self.evaluate_w, [0])) + list(product([0], self.evaluate_w))
        else:
            self.congater_names = None
        self.training_stage = ["task"] + self.congater_names if self.congater_names else ["task"]

        self.val_w = [0, 1]
        self.default_trainable_parameters = default_trainable_parameters
        self.num_encoder_layers = len(self.encoder.transformer.layer)

        #add Congater layers
        if self.congater_names:
            self.congater = nn.ModuleDict()
            for name in self.congater_names:
                self.congater[name] = nn.ModuleList()
                if self.congater_position == "all":
                    for i in range(self.num_encoder_layers):
                        self.congater[name].append(GateLayer(self.hidden_size, num_layers=num_gate_layers,
                                                             squeeze_ratio=gate_squeeze_ratio, version=congater_version))
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
                                                         squeeze_ratio=gate_squeeze_ratio, version=congater_version))

        # bottleneck layer
        if self.has_bottleneck:
            self.bottleneck = ClfHead(self.hidden_size, bottleneck_dim, dropout=bottleneck_dropout)
            self.in_size_heads = bottleneck_dim
        else:
            self.bottleneck = torch.nn.Identity()
            self.in_size_heads = self.hidden_size

        #set trainable parameters
        self.set_trainable_parameters(self.default_trainable_parameters)

    def set_trainable_parameters(self, trainable_param):
        ln_gate = False
        if trainable_param == "none":
            for param in self.parameters():
                param.requires_grad = False
        elif trainable_param == 'only_output':
            for name, param in self.named_parameters():
                if ('output' in name) and ('attention' not in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif trainable_param == 'all':
            for param in self.parameters():
                param.requires_grad = True
        else:
            for name, param in self.named_parameters():
                if self.condition_(trainable_param, name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

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

    def forward_gate(self, x, w, layer):
        x_gate = []
        for name in self.congater_names:
            x_gate.append(self.congater[name][layer](x, w[name]))
        if len(x_gate) > 1:
            if self.congater_version == 1:
                x_gate = [x_gate[i] * x_gate[i - 1] for i in range(1, len(x_gate))][0]
            elif self.congater_version == 2:
                x_gate = [x_gate[i] + x_gate[i - 1] for i in range(1, len(x_gate))][0]
            else:
                print("Version not Recognized")
        else:
            x_gate = x_gate[0]

        return x_gate

    def _forward(self, w, **x) -> torch.Tensor:
        if self.congater_position == "all":
            embed = self.encoder.embeddings(x["input_ids"])
            x = {"x": embed, "attn_mask": x["attention_mask"]}
            for i in range(len(self.encoder.transformer.layer)):
                if self.congater_version == 1:
                    x["x"] = self.encoder.transformer.layer[i](**x)[0]*self.forward_gate(x["x"], w, i)
                elif self.congater_version == 2:
                    x["x"] = self.encoder.transformer.layer[i](**x)[0]+self.forward_gate(x["x"], w, i)

            hidden = x["x"]
        else: 
            hidden = self.encoder(**x)
            if self.congater_version == 1:
                hidden = hidden * self.forward_gate(hidden, w, -1)
            else:
                hidden = hidden + self.forward_gate(hidden, w, -1)

        return self.bottleneck(hidden)

    #init for parameters needed for training
    def init_training_par(self, global_step, best_values, best_steps, running_metrics, val_metrics, train_loss, logging_loss, best_metrics):
        self.global_step = global_step
        self.best_values = best_values
        self.best_steps = best_steps
        self.running_metrics = running_metrics
        self.val_metrics = val_metrics
        self.train_loss = train_loss
        self.logging_loss = logging_loss
        self.best_metrics = best_metrics

    #forward function
    def forward(self, query_token_ids: torch.Tensor, query_mask: torch.Tensor = None, doc_emb: torch.Tensor = None,
                docinds: torch.Tensor = None, local_emb_mat: torch.Tensor = None, doc_padding_mask: torch.Tensor = None,
                doc_attention_mat_mask: torch.Tensor = None, doc_neutscore: torch.Tensor = None, labels: torch.Tensor = None,
                w: int = None, bias_regul_coeff: float = None) -> Dict[str, torch.Tensor]:
        r"""
        num_docs is the number of candidate docs per query and corresponds to the length of the padded "decoder" sequence
        :param  query_token_ids: (batch_size, max_query_len) tensor of padded sequence of token IDs fed to the encoder
        :param  query_mask: (batch_size, query_length) attention mask bool tensor for query tokens; 0 ignore, non-0 use
        :param  doc_emb: (batch_size, num_docs, doc_emb_dim) sequence of document embeddings fed to the "decoder".
                    Mutually exclusive with `docinds`.
        :param  docinds: (batch_size, num_docs) tensor of local indices of documents corresponding to rows of the
                    `local_emb_mat` used to lookup document vectors in nn.Embedding. Mutually exclusive with `doc_emb`.
        :param  local_emb_mat: (num_unique_docIDs, doc_emb_dim) tensor of local doc embedding matrix containing emb. vectors
                    of all unique documents in the batch.  Used with `docinds` to lookup document vectors in nn.Embedding on the GPU.
                    This is done to avoid replicating embedding vectors of in-batch negatives, thus sparing GPU bandwidth.
                    Global matrix cannot be used, because the collection size is in the order of 10M: GPU memory!
        :param  doc_padding_mask: (batch_size, num_docs) boolean/ByteTensor mask with 0 at positions of missing input
                    documents (decoder sequence length is less than the max. doc. pool size in the batch)
        :param  doc_attention_mat_mask: (num_docs, num_docs) float additive mask for the decoder sequence (optional).
                    This is for causality, and if FloatTensor, can be directly added on top of the attention matrix.
                    If BoolTensor, positions with ``True`` are ignored, while ``False`` values will be considered.
        :param  doc_neutscore: (batch_size, num_docs) sequence of document neutrality scores to calculate fairness.
        :param  labels: (batch_size, num_docs) int tensor which for each query (row) contains the indices of the
                relevant documents within its corresponding pool of candidates (docinds).
                    Optional: If provided, the loss will be computed.

        :returns:
            dict containing:
                rel_scores: (batch_size, num_docs) relevance scores in [0, 1]
                loss: scalar mean loss over entire batch (only if `labels` is provided!)
        """
        if doc_emb is None:  # happens only in training, when additionally there is in-batch negative sampling
            doc_emb = self.lookup_doc_emb(docinds, local_emb_mat)  # (batch_size, max_docs_per_query, doc_emb_dim)
            
        if self.project_documents is not None:
            doc_emb = self.project_documents(doc_emb)
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]
        doc_emb = doc_emb.permute(1, 0, 2)  # (max_docs_per_query, batch_size, doc_emb_dim) document embeddings

        if query_token_ids.size(0) != doc_emb.size(1):
            raise RuntimeError("the batch size for queries and documents must be equal")
        
        x = {'input_ids': query_token_ids.to(torch.int64), 'attention_mask': query_mask}

        # Changed model call here to Congater _forward method
        encoder_out = self._forward(w, **x)  # int64 required by torch nn.Embedding
        enc_hidden_states = encoder_out#['last_hidden_state']  # (batch_size, max_query_len, query_dim), torch.Size([5, 11, 768])
        if self.query_dim != self.d_model:  # project query representation vectors to match dimensionality of doc embeddings
            enc_hidden_states = self.project_query(enc_hidden_states)  # (batch_size, max_query_len, d_model)
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]
        enc_hidden_states = enc_hidden_states.permute(1, 0, 2)  # (max_query_len, batch_size, d_model)

        # The nn.MultiHeadAttention expects ByteTensor or Boolean and uses the convention that non-0 is ignored
        # and 0 is used in attention, which is the opposite of HuggingFace.
        memory_key_padding_mask = ~query_mask

        if self.no_decoder:
            output_emb = doc_emb
        else:
            if self.selfatten_mode == 1:  # NOTE: for ablation study. Turn off SA by using diagonal SA matrix (no interactions between documents)
                doc_attention_mat_mask = ~torch.eye(doc_emb.shape[0], dtype=bool).to(device=doc_emb.device)  # (max_docs_per_query, max_docs_per_query)
            # (num_docs, batch_size, doc_emb_size) transformed sequence of document embeddings
            output_emb = self.decoder(doc_emb, enc_hidden_states, tgt_mask=doc_attention_mat_mask,
                                      tgt_key_padding_mask=~doc_padding_mask,  # again, MultiHeadAttention opposite of HF
                                      memory_key_padding_mask=memory_key_padding_mask)
            # output_emb = self.act(output_emb)  # the output transformer encoder/decoder embeddings don't include non-linearity

        predictions = self.score_docs(output_emb, enc_hidden_states, memory_key_padding_mask)  # relevance scores. dimensions vary depending on scoring_mode
        
        if self.scoring_mode.endswith('softmax'):
            rel_scores = torch.exp(predictions[:, :, 0])  # (batch_size, num_docs) relevance scores
        else:
            rel_scores = predictions.squeeze()  # (batch_size, num_docs) relevance scores
            
        # Fairness regularization term  # TODO: wrap in a separate function
        bias_regul_term = None
        if doc_neutscore is not None:

            _cutoff = np.min([self.bias_regul_cutoff, doc_neutscore.shape[1]])

            _indices_sorted = torch.argsort(rel_scores, dim=1, descending=True)
            _indices_sorted[_indices_sorted < _cutoff] = -1
            _indices_sorted[_indices_sorted != -1] = 0
            _indices_sorted[_indices_sorted == -1] = 1
            _indices_mask = doc_neutscore.new_zeros(doc_neutscore.shape)    
            _indices_mask[_indices_sorted == 0] = float("-Inf")

            doc_neutscore_probs = torch.nn.Softmax(dim=1)(doc_neutscore + _indices_mask)
            rel_scores_logprobs = torch.nn.LogSoftmax(dim=1)(rel_scores + _indices_mask)

            bias_regul_term = torch.nn.KLDivLoss(reduction='batchmean')(rel_scores_logprobs, doc_neutscore_probs)

        # Compute loss
        if labels is not None:
            loss = self.loss_module(rel_scores, labels.to(torch.int64))  # loss is scalar tensor. labels are int16, convert to int64 for PyTorch losses

            if self.aux_loss_module is not None and (self.aux_loss_coeff > 0):  # add auxiliary loss, if specified
                loss += self.aux_loss_coeff * self.aux_loss_module(rel_scores, labels.to(torch.int64))
            if bias_regul_term is not None:
                if bias_regul_coeff < 0:
                    loss = rel_scores.new([0])[0]
                    loss += - bias_regul_coeff * bias_regul_term
                else:    
                    loss += bias_regul_coeff * bias_regul_term
            
            return {'loss': loss, 'rel_scores': rel_scores}
        return {'rel_scores': rel_scores}



    @torch.no_grad()
    def evaluate(self, args, dataloader, stage, bias_regul_coeff, logger, fairrmetric=None):
        """
        Wrapper for the evaluation function. The parameter w is set according to the given stage
        :return:
            eval_metrics: dict containing metrics (at least 1, batch processing time)
            rank_df: dataframe with indexed by qID (shared by multiple rows) and columns: PID, rank, score
            w: parameter w that was used during evaluation
        """

        #set w
        w = dict(zip(self.congater_names, [0 for _ in self.congater_names])) if self.congater_names else 0

        #set w and loss debiasing parameter according to stage
        if stage != "task":
            _bias_regul_coeff = bias_regul_coeff
            w[f"{stage}"] = 1
        else:
            _bias_regul_coeff = 0

        eval_metrics, ranked_df = self.eval_step(args, dataloader, _bias_regul_coeff, w, logger, fairrmetric)

        return eval_metrics, ranked_df, w


    def eval_step(self, args, dataloader, bias_regul_coeff, w, logger, fairrmetric=None):
        """
        Evaluate the model on the dataset contained in the given dataloader and compile a dataframe with
        document ranks and scores for each query. If the dataset includes relevance labels (qrels), then metrics
        such as MRR, MAP etc will be additionally computed.
        Stage parameter for congater is not needed as evaluation is the same for all stages.
        :return:
            eval_metrics: dict containing metrics (at least 1, batch processing time)
            rank_df: dataframe with indexed by qID (shared by multiple rows) and columns: PID, rank, score
        """
        qrels = dataloader.dataset.qrels  # dict{qID: dict{pID: relevance}}
        labels_exist = qrels is not None

        # num_docs is the (potentially variable) number of candidates per query
        relevances = []  # (total_num_queries) list of (num_docs) lists with non-zeros at the indices corresponding to actually relevant passages
        num_relevant = []  # (total_num_queries) list of number of ground truth relevant documents per query
        df_chunks = []  # (total_num_queries) list of dataframes, each with index a single qID and corresponding (num_docs) columns PID, rank, score
        query_time = 0  # average time for the model to score candidates for a single query
        total_loss = 0  # total loss over dataset

        with torch.no_grad():
            for batch_data, qids, docids in tqdm(dataloader, desc="Evaluating"):
                batch_data = {k: v.to(args.device) for k, v in batch_data.items()}
                start_time = time.perf_counter()
                try:
                    out = self.forward(**batch_data, w = w, bias_regul_coeff = bias_regul_coeff)
                except RuntimeError:
                    raise optuna.exceptions.TrialPruned()
                query_time += time.perf_counter() - start_time
                rel_scores = out['rel_scores'].detach().cpu().numpy()  # (batch_size, num_docs) relevance scores in [0, 1]
                if 'loss' in out:
                    total_loss += out['loss'].sum().item()
                assert len(qids) == len(docids) == len(rel_scores)

                # Rank documents based on their scores
                num_docs_per_query = [len(cands) for cands in docids]
                num_lengths = set(num_docs_per_query)
                no_padding = (len(num_lengths) == 1)  # whether all queries in this batch had the same number of candidates

                if no_padding:  # (only) 10% speedup compared to other case
                    docids_array = np.array(docids, dtype=np.int32)  # (batch_size, num_docs) array of docIDs per query
                    # First shuffle along doc dimension, because relevant document(s) are placed at the beginning and would benefit
                    # in case of score ties! (can happen e.g. with saturating score functions)
                    inds = np.random.permutation(rel_scores.shape[1])
                    np.take(rel_scores, inds, axis=1, out=rel_scores)
                    np.take(docids_array, inds, axis=1, out=docids_array)

                    # Sort by descending relevance
                    inds = np.fliplr(np.argsort(rel_scores, axis=1))  # (batch_size, num_docs) inds to sort rel_scores
                    # (batch_size, num_docs) docIDs per query, in order of descending relevance score
                    ranksorted_docs = np.take_along_axis(docids_array, inds, axis=1)
                    sorted_scores = np.take_along_axis(rel_scores, inds, axis=1)
                else:
                    # (batch_size) iterables of docIDs and scores per query, in order of descending relevance score
                    ranksorted_docs, sorted_scores = zip(*(map(rank_docs, docids, rel_scores)))

                # extend by batch_size elements
                df_chunks.extend(pd.DataFrame(data={"PID": ranksorted_docs[i],
                                                    "rank": list(range(1, len(docids[i]) + 1)),
                                                    "score": sorted_scores[i]},
                                            index=[qids[i]] * len(docids[i])) for i in range(len(qids)))

                if labels_exist:
                    relevances.extend(get_relevances(qrels[qids[i]], ranksorted_docs[i]) for i in range(len(qids)))
                    num_relevant.extend(len([docid for docid in qrels[qid] if qrels[qid][docid] > 0]) for qid in qids)

        if labels_exist:
            try:
                eval_metrics = calculate_metrics(relevances, num_relevant, args.metrics_k)  # aggr. metrics for the entire dataset
            except:
                logger.error('Metrics calculation failed!')
                eval_metrics = OrderedDict()
            eval_metrics['loss'] = total_loss / len(dataloader.dataset)  # average over samples
        else:
            eval_metrics = OrderedDict()
        eval_metrics['query_time'] = query_time / len(dataloader.dataset)  # average over samples
        ranked_df = pd.concat(df_chunks, copy=False)  # index: qID (shared by multiple rows), columns: PID, rank, score

        # Evaluate fairness  # TODO: split into separate function
        if fairrmetric is not None:
            try:
                _retrievalresults = {}
                for _item in df_chunks:
                    _qid = _item.index[0]
                    _retrievalresults[_qid] = _item['PID'].tolist()

                eval_FaiRR, eval_NFaiRR = fairrmetric.calc_FaiRR_retrievalresults(retrievalresults=_retrievalresults)

                for _cutoff in eval_FaiRR:
                    eval_metrics['NFaiRR_cutoff_%d' % _cutoff] = eval_NFaiRR[_cutoff]
            except:
                logger.error('Fairness metrics calculation failed!')

        if labels_exist and (args.debug or args.task != 'train'):
            try:
                rs = (np.nonzero(r)[0] for r in relevances)
                ranks = [1 + int(r[0]) if r.size else 1e10 for r in rs]  # for each query, what was the rank of the rel. doc
                freqs, bin_edges = np.histogram(ranks, bins=[1, 5, 10, 20, 30] + list(range(50, 1050, 50)))
                bin_labels = ["[{}, {})".format(bin_edges[i], bin_edges[i + 1])
                            for i in range(len(bin_edges) - 1)] + ["[{}, inf)".format(bin_edges[-1])]
                logger.info('\nHistogram of ranks for the ground truth documents:\n')
                ascii_bar_plot(bin_labels, freqs, width=50, logger=logger)
            except:
                logger.error('Not possible!')

        return eval_metrics, ranked_df


    def validate(self, args, val_dataloader, tensorboard_writer, training_stages, bias_regul_coeff, logger, fairrmetric=None):
        """Run an evaluation on the validation set while logging metrics, and handle result"""

        #iterate over training stages
        for label_idx, stage in enumerate(training_stages):
            #For training_method post this function is only called per training stage, for other cases code would have to be adapted
            assert len(training_stages)==1, "Training_method par is not supported."

            #call evaluation function
            self.eval()
            eval_start_time = time.time()
            val_metrics, ranked_df, w = self.evaluate(args, val_dataloader, stage, bias_regul_coeff, logger, fairrmetric=fairrmetric)
            eval_runtime = time.time() - eval_start_time
            self.train()
            logger.info("Stage {}: Validation runtime: {} hours, {} minutes, {} seconds\n".format(stage, *readable_time(eval_runtime)))

            #output times to logger
            global val_times
            val_times.update(eval_runtime)
            avg_val_time = val_times.get_average()
            avg_val_batch_time = avg_val_time / len(val_dataloader)
            avg_val_sample_time = avg_val_time / len(val_dataloader.dataset)
            logger.info("Stage {}: Avg val. time: {} hours, {} minutes, {} seconds".format(stage, *readable_time(avg_val_time)))
            logger.info("Stage {}: Avg batch val. time: {} seconds".format(stage, avg_val_batch_time))
            logger.info("Stage {}: Avg sample val. time: {} seconds".format(stage, avg_val_sample_time))

            #output results to tensorboard
            print_str = 'Stage {}: Step {} Validation Summary: '.format(stage, self.global_step)
            for k, v in val_metrics.items():
                tensorboard_writer.add_scalar('dev/{}/{}'.format(stage,k), v, self.global_step)
                print_str += '{}: {:8f} | '.format(k, v)
            logger.info(print_str)

            #code determines the best step according to metric: one key_metric for both training stages, it is better to use the last checkpoints per stage
            val_metrics["global_step"] = self.global_step
            if args.key_metric == 'F1_fairness':
                metric_value = 2 * val_metrics['MRR@10'] * val_metrics['NFaiRR_cutoff_10']/(val_metrics['MRR@10'] + val_metrics['NFaiRR_cutoff_10'])
            else:
                metric_value = val_metrics[args.key_metric]
            if args.key_metric in NEG_METRICS:
                ind = bisect.bisect_right(self.best_values, metric_value)  # index where to insert in sorted list in ascending order
            else:
                ind = reverse_bisect_right(self.best_values, metric_value)  # index where to insert in sorted list in descending order
            condition = (ind < args.num_keep_best) and (self.global_step > STEP_THRESHOLD)  # NOTE: the second condition is because fairness is always bad initially but performance at its best
            if condition:
                self.best_values.insert(ind, metric_value)
                self.best_steps.insert(ind, self.global_step)
                # to save space: optimizer, scheduler not saved! Only latest checkpoints can be used for resuming training perfectly
                save_model(os.path.join(args.save_dir, stage, 'model_best_{}.pth'.format(self.global_step)), self.global_step, self)
                if len(self.best_values) > args.num_keep_best:
                    os.remove(os.path.join(args.save_dir, stage, 'model_best_{}.pth'.format(self.best_steps[-1])))
                    self.best_values = self.best_values[:args.num_keep_best]
                    self.best_steps = self.best_steps[:args.num_keep_best]

                if not args.no_predictions:
                    ranked_filepath = os.path.join(args.pred_dir, stage, 'best.ranked.dev.tsv')
                    ranked_df.to_csv(ranked_filepath, header=False, sep='\t')

                # Export metrics to a file accumulating best records from the current experiment
                rec_filepath = os.path.join(args.pred_dir, stage, 'training_session_records.xls')
                register_record(rec_filepath, args.initial_timestamp, args.experiment_name, val_metrics, w = w[self.congater_names[0]])

        return val_metrics


    def _step(
        self,
        args: dict,
        train_loader: DataLoader,
        val_dataloader: DataLoader,
        logger: logging.Logger,
        tb_writer: SummaryWriter,
        training_stages: list,
        bias_regul_coeff: float,
        batch_times: Timer,
        total_training_steps: int,
        fairrmetric=None,
        trial = None

    ) -> None:
        """
        Step function handles main part of training logic: set trainable parameters accoring to stage, make the training step, logging and validation, save checkpoint.
        :return:
            step: return actual training step
        """
        self.train()
        sstr = ['non-encoder', 'encoder']
        epoch_str = "training - step {}, loss: {:7.5f}"
        batch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan), leave=False, position=1)

        #iterate over all batches
        for step, (model_inp, _, _) in enumerate(batch_iterator):
            model_inp = {k: v.to(args.device) for k, v in model_inp.items()}
            for idx, stage in enumerate(training_stages):
                w = dict(zip(self.congater_names, [0 for _ in self.congater_names])) if self.congater_names else 0
                if stage == "task":
                    self.set_trainable_parameters(self.default_trainable_parameters)
                else:
                    if args.congater_training_method == "post":
                        self.set_trainable_parameters(f"{stage}")
                    elif args.congater_training_method == "par": #other training_methods can be applied but are not implemented at this point
                        raise NotImplementedError
                    w[f"{stage}"] = 1

                loss = 0.
                start_time = time.perf_counter()

                #if stage is 'task' set debiasing loss to zero, otherwise keep input parameter
                if stage != "task" and self.congater_names:
                    _bias_regul_coeff = bias_regul_coeff
                else:
                    _bias_regul_coeff = 0

                #updated forward pass
                try:
                    output = self.forward(**model_inp, w = w, bias_regul_coeff = _bias_regul_coeff)  # model output is a dictionary
                except RuntimeError:
                    raise optuna.exceptions.TrialPruned()
                loss = output['loss']

                #Directly from Coder training
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                if args.grad_accum_steps > 1:
                    loss = loss / args.grad_accum_steps
                loss.backward()  # calculate gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), args.max_grad_norm)
                self.train_loss += loss.item()

                batch_times.update(time.perf_counter() - start_time)

                if (step + 1) % args.grad_accum_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()  # Update learning rate schedule
                    self.zero_grad()
                    self.global_step += 1


                    # logging for training
                    if args.logging_steps and (self.global_step % args.logging_steps == 0):
                        for s in range(len(self.scheduler.schedulers)):
                            # first brackets select scheduler, second the group
                            tb_writer.add_scalar('learn_rate{}/{}'.format(s, stage), self.scheduler.get_last_lr()[s][0], self.global_step)
                        cur_loss = (self.train_loss - self.logging_loss) / args.logging_steps  # mean loss over last args.logging_steps (smoothened "current loss")
                        tb_writer.add_scalar('train/{}/loss'.format(stage), cur_loss, self.global_step)

                        if args.debug:
                            logger.debug("Mean loss over {} steps: {:.5f}".format(args.logging_steps, cur_loss))
                            for i, s in enumerate(self.scheduler.get_last_lr()):
                                logger.debug('Learning rate ({}): {}'.format(sstr[i], s))
                            logger.debug("Current memory usage: {} MB or {} MB".format(np.round(get_current_memory_usage()),
                                                                            np.round(get_current_memory_usage2())))
                            logger.debug("Max memory usage: {} MB".format(int(np.ceil(get_max_memory_usage()))))

                            logger.debug("Average lookup time: {} s /samp".format(lookup_times.get_average()))
                            logger.debug("Average retr. candidates time: {} s /samp".format(retrieve_candidates_times.get_average()))
                            logger.debug("Average prep. docids time: {} s /samp".format(prep_docids_times.get_average()))
                            logger.debug("Average sample fetching time: {} s /samp".format(sample_fetching_times.get_average()))
                            logger.debug("Average collation time: {} s /batch".format(collation_times.get_average()))
                            logger.debug("Average total batch processing time: {} s /batch".format(batch_times.get_average()))

                            # logger.debug("Score parameters: {}".format(score_params))  # TODO: DEBUG

                        self.logging_loss = self.train_loss

                    # evaluate at specified interval or if this is the last step
                    if (args.validation_steps and (self.global_step % args.validation_steps == 0)) or self.global_step == total_training_steps:

                        logger.info("\n\n***** Running evaluation of step {} on dev set *****".format(self.global_step))
                        self.val_metrics = self.validate(args, val_dataloader, tb_writer,
                                                                        [stage], self.bias_regul_coeff, logger, fairrmetric=fairrmetric)
                        if len(self.best_steps) and (self.best_steps[0] == self.global_step):
                            self.best_metrics = self.val_metrics.copy()
                        self.metrics_names, self.metrics_values = zip(*self.val_metrics.items())
                        self.running_metrics.append(list(self.metrics_values))

                        if args.reduce_on_plateau:
                            self.ROP_scheduler.step(self.val_metrics[args.reduce_on_plateau])

                        if trial is not None:  # used for hyperparameter optimization
                            trial.report(self.best_metrics[args.key_metric], self.global_step)
                            HARD_PATIENCE = 60000
                            HARD_TOLERANCE = 0.001
                            if len(self.best_steps) and ((self.global_step - self.best_steps[-1]) > HARD_PATIENCE):  # countdown
                            #if (global_step > HARD_PATIENCE) and (running_metrics[0][metric2ind[args.key_metric]] - best_metrics[args.key_metric]) < HARD_TOLERANCE):
                                return step

                    if (args.save_steps and (self.global_step % args.save_steps == 0)) or self.global_step == total_training_steps or self.global_step == int(total_training_steps/len(self.training_stage)):
                        # Save model checkpoint
                        save_model(os.path.join(args.save_dir, stage, 'model_{}.pth'.format(self.global_step)),
                                        self.global_step, self, self.optimizer, self.scheduler)
                        remove_oldest_checkpoint(os.path.join(args.save_dir, stage), args.num_keep)
        return step


    def _init_optimizer_and_schedule(
        self,
        args: dict,
        logger: logging.Logger,
        start_step: int,
        optim_state: dict,
        total_training_steps: int,
        sched_state: dict
    ) -> None:
        """
        Initialize optimizer and scheduler for training
        """
        
        # Prepare optimizer and schedule
        nonencoder_optimizer, encoder_optimizer = get_optimizers(args, self)
        logger.debug("args.learning_rate: {}".format(args.learning_rate))
        logger.debug('nonencoder_optimizer.defaults["lr"]: {}'.format(nonencoder_optimizer.defaults["lr"]))
        optimizer = MultiOptimizer(nonencoder_optimizer)
        if args.encoder_delay <= start_step:
            optimizer.add_optimizer(encoder_optimizer)
        if optim_state is not None:
            optimizer.load_state_dict(optim_state)
            logger.info('Loaded optimizer(s) state')
            logger.debug('optimizer.defaults["lr"]: {}'.format(optimizer.optimizers[0].defaults["lr"]))
        self.optimizer = optimizer

        schedulers = get_schedulers(args, total_training_steps, nonencoder_optimizer, encoder_optimizer)

        logger.debug("schedulers['nonencoder_scheduler'].get_last_lr(): {}".format(schedulers['nonencoder_scheduler'].get_last_lr()))
        scheduler = MultiScheduler(schedulers['nonencoder_scheduler'])
        if args.reduce_on_plateau:
            ROP_scheduler = MultiScheduler(schedulers['ROP_nonencoder_scheduler'])
        if args.encoder_delay <= start_step:
            scheduler.add_scheduler(schedulers['encoder_scheduler'])
            if args.reduce_on_plateau:
                ROP_scheduler.add_scheduler(schedulers['ROP_encoder_scheduler'])
        if sched_state is not None:
            scheduler.load_state_dict(sched_state)
            logger.info('Loaded scheduler(s) state')
        scheduler.step()  # this is done to correctly initialize learning rate (otherwise optimizer.defaults["lr"] is the first value when using get_constant_schedule_with_warmup)
        logger.debug("schedulers['nonencoder_scheduler'].get_last_lr(): {}".format(scheduler.schedulers[0].get_last_lr()))

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ROP_scheduler = ROP_scheduler