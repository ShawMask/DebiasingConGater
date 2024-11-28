import pickle
import pandas as pd

with open( '/mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/data/bios/new_bios.pkl', 'rb') as file:
    dicts = pickle.load(file)
bios = pd.DataFrame(dicts).reset_index()
bios = bios.drop('index', axis=1)


with open( '/mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/data/bios/test.pkl', 'rb') as file:
    dicts = pickle.load(file)
uk_jobs_test = pd.DataFrame(dicts)

test_hits = pd.read_csv('/mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/data/bios/run.bios.bm25.top100.test.tsv', sep=' ', header=None, names=['query_id', 'Q0', 'doc_id', 'rank', 'score','Anserini'])
test_hits = test_hits.drop(['Q0','rank','score','Anserini'], axis='columns')
test_hits['doc_id'] = test_hits['doc_id'].replace('doc','',regex=True).astype(int)

from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util
from sentence_transformers import SentenceTransformer
import torch


base_path = '/mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/coderdebiasinf/results/bi-encoder/'
model_name_list = ['bi-encoder-distilbert-base-uncased-2023-08-10_15-51-20-latest'] #['cross-encoder-bert-base-uncased-2023-05-06_14-22-10-latest']
for model_name in model_name_list:

    model = SentenceTransformer(base_path+model_name, device=torch.device("cuda:0"))

    from tqdm.autonotebook import tqdm
    import numpy as np

    result = []

    for id in tqdm(range(len(uk_jobs_test))):
        new_bios = bios.loc[test_hits[test_hits['query_id']==id]['doc_id']]
        query = uk_jobs_test['description'][id]
        corpus_embedding = model.encode(list(new_bios['raw']))
        query_embedding = model.encode(query)
        bi_scores = util.semantic_search(query_embedding, corpus_embedding, top_k=10)
        bi_scores = sorted(bi_scores[0], key=lambda x: x['score'], reverse=True)
        bi_hits = [hits['corpus_id'] for hits in bi_scores]
        bi_scores = [hits['score'] for hits in bi_scores]
    
        result.append({'corpus_id': new_bios.index[np.flip(np.argsort(bi_scores))],'scores':sorted(bi_scores, reverse=True)})

    import pickle

    with open('/mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/coderdebiasinf/results/bi-encoder/'+ model_name +'.pkl', 'wb') as f:
        pickle.dump(result, f)