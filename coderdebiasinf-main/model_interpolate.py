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
from transformers.adapters import PfeifferConfig
import transformers.adapters.composition as ac
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.adapters.composition import adjust_tensors_for_parallel
from transformers.adapters.context import AdapterSetup, ForwardContext
import transformers.adapters as adp
adp.__file__

val_times = Timer()  # stores measured validation times

STEP_THRESHOLD = 0  #4000 # Used for fairness regularization; checkpoints corresponding to best performance before STEP_THRESHOLD steps will be ignored
    


class AdapterInterpolateModel(MDSTransformer):

    def __init__(
        self,
        bottleneck: bool = False,
        bottleneck_dim: Optional[int] = None,
        bottleneck_dropout: Optional[float] = None,
        adapter_names: Optional[str] = "gender",
        interpolate_position: str = "all",
        squeeze_ratio: int = 4,
        custom_init: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)


        self.has_bottleneck = bottleneck
        self.bottleneck_dim = bottleneck_dim
        self.bottleneck_dropout = bottleneck_dropout
        self.squeeze_ratio = squeeze_ratio
        self.global_step = 0
        self.custom_init = custom_init
        self.adapter_names = adapter_names
        self.interpolate_position = interpolate_position

        self.training_stage = [self.adapter_names]
        self.num_encoder_layers = len(self.encoder.transformer.layer)

        adap_config = PfeifferConfig(reduction_factor=self.squeeze_ratio)
        self.encoder.add_adapter("task", adap_config)
        self.encoder.add_adapter(self.adapter_names, adap_config)
        self.encoder.active_adapters = ac.Parallel("task", self.adapter_names)
        self.config = self.encoder.config
        print(self.encoder.active_adapters)
        self.num_encoder_layers = len(self.encoder.transformer.layer)

        # bottleneck layer
        if self.has_bottleneck:
            self.bottleneck = ClfHead(self.hidden_size, bottleneck_dim, dropout=bottleneck_dropout)
            self.in_size_heads = bottleneck_dim
        else:
            self.bottleneck = torch.nn.Identity()
            self.in_size_heads = self.hidden_size

    def print_trainable_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)

    # def _forward(self, w, **x) -> torch.Tensor:
    #     self.encoder.active_adapters = ac.Parallel("task", self.adapter_names)
    #     #get normal layer output
    #     hidden_enc = self.encoder(**x)
    #     hidden_enc = hidden_enc['last_hidden_state']
    #     #combine for inference
    #     hidden = (1-w[self.adapter_names]) * hidden_enc[:hidden_enc.size(0) // 2] + w[self.adapter_names] * hidden_enc[hidden_enc.size(0) // 2:]
    #     return self.bottleneck(hidden)

    @ForwardContext.wrap
    def _forward(self, w, head_mask=None, **x) -> torch.Tensor:
        if "all" in self.interpolate_position:
            #is needed for Distilbert
            input_shape = x["input_ids"].size()
            # past_key_values_length: not needed in distilbert
            #past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

            #is not needed at all in Distilbert
            # if x["token_type_ids"] is None:
            #     if hasattr(self.encoder.embeddings, "token_type_ids"):
            #         buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
            #         buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
            #         token_type_ids = buffered_token_type_ids_expanded
            #     else:
            #         token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.device)

            ##is not needed at all in Distilbert
            # extended_attention_mask: torch.Tensor = self.encoder.get_extended_attention_mask(x["attention_mask"],
            #                                                                                  input_shape)
            #new for distilbert! --> not needed, attention_mask comes from x
            if x['attention_mask'] is None:
               x['attention_mask'] = torch.ones(input_shape, device=self.device)
            
            #Is needed in Distilbert
            head_mask = self.encoder.get_head_mask(head_mask, self.config.num_hidden_layers)

            embedding_output = self.encoder.embeddings(
                input_ids=x["input_ids"]
                #Is not needed for distilbert
                # token_type_ids=x["token_type_ids"],
                # past_key_values_length=past_key_values_length,
            )

            embedding_output = self.encoder.invertible_adapters_forward(embedding_output)
            x = {"x": embedding_output,
                 "attn_mask": x["attention_mask"],
                 "layer_wise": True}

            for i, layer_module in enumerate(self.encoder.transformer.layer):
                self.encoder.active_adapters = ac.Parallel("task", self.adapter_names)
                #Is ok for distilbert
                x["head_mask"] = head_mask[i]
                org_hidden = layer_module(**x)[-1]
                #is ok for distilbert
                (x["attn_mask"],) = adjust_tensors_for_parallel(x["x"], x["attn_mask"])
                x["x"] = (1-w[self.adapter_names]) * org_hidden[:org_hidden.size(0) // 2] + w[self.adapter_names] * org_hidden[org_hidden.size(0) // 2:]
            hidden = x["x"]
            #Is not done in distilbert
            #hidden = x["hidden_states"][:, 0]
        else:
            self.encoder.active_adapters = ac.Parallel("task", self.adapter_names)
            org_hidden = self.encoder(**x)
            org_hidden = org_hidden['last_hidden_state']
            hidden = (1-w[self.adapter_names]) * org_hidden[:org_hidden.size(0) // 2] + w[self.adapter_names] * org_hidden[org_hidden.size(0) // 2:]

        #not needed in Test part
        #self.encoder.train_adapter("task") if w == 0 else self.encoder.train_adapter(self.adapter_names)
        return self.bottleneck(hidden)
    

    # def _forward(self, w, **x) -> torch.Tensor:
    #     #get normal layer output
    #     embed = self.encoder.embeddings(x["input_ids"])
    #     x = {"x": embed, "attn_mask": x["attention_mask"]}
    #     for i in range(len(self.encoder.transformer.layer)):
    #         self.encoder.active_adapters = None
    #         sa_output = self.encoder.transformer.layer[i].attention(
    #         query=x["x"],
    #         key=x["x"],
    #         value=x["x"],
    #         mask=x["attn_mask"],
    #         head_mask=None, #Check if model has head_mask?
    #         output_attentions=None, #does not have output_attentions according to config
    #         )
    #         sa_output = sa_output[0]
    #         sa_output = self.encoder.transformer.layer[i].attention_adapters(sa_output, x["x"], self.encoder.transformer.layer[i].sa_layer_norm)
    #         ffn_output = self.encoder.transformer.layer[i].ffn(sa_output)
    #         self.encoder.set_active_adapters(self.adapter_names)
    #         adapter_output = self.encoder.transformer.layer[i].output_adapters(ffn_output, sa_output, self.encoder.transformer.layer[i].output_layer_norm)[0]
    #         hidden = (1-w[self.adapter_names]) * ffn_output[0] + w[self.adapter_names] * adapter_output
    #         x["x"] = hidden
    #     return self.bottleneck(hidden)

    #TODO: evaluate if this is not a training parameter
    def init_training_par(self, global_step, best_values, best_steps, running_metrics, val_metrics, train_loss, logging_loss, best_metrics):
        self.global_step = global_step
        self.best_values = best_values
        self.best_steps = best_steps
        self.running_metrics = running_metrics
        self.val_metrics = val_metrics
        self.train_loss = train_loss
        self.logging_loss = logging_loss
        self.best_metrics = best_metrics
        #self.metrics_names, self.metrics_values = zip(*self.val_metrics.items())


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

        # Changed model call here to forward
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
    def eval_step(self, args, dataloader, bias_regul_coeff, w, logger, fairrmetric=None):
        """
        Evaluate the model on the dataset contained in the given dataloader and compile a dataframe with
        document ranks and scores for each query. If the dataset includes relevance labels (qrels), then metrics
        such as MRR, MAP etc will be additionally computed.
        Stage for congater is not needed as evaluation is the same for all stages.
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

