

import pandas as pd
import pickle
import tqdm
import numpy as np


#save documents tsv
df_doc = pd.read_csv('/mnt/c/users/cornelia/documents/AI/MasterThesis/IRDebias/data/bios/new_bios.tsv', sep="\t")
df_doc = df_doc.rename({'Unnamed: 0': 'Id'}, axis=1)
df_doc[["Id","bio"]].to_csv('/mnt/c/users/cornelia/documents/AI/MasterThesis/IRDebias/data/bios/documents.tsv', sep="\t", index=False,header=False)

df_doc['female_score'] = np.where(df_doc['gender']=='F', 1, 0)
df_doc[["Id","female_score"]].to_csv('/mnt/c/users/cornelia/documents/AI/MasterThesis/IRDebias/data/bios/collection_neutralityscores_direct.tsv', sep="\t", index=False,header=False)

#save query tsv:
with open( '/mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/data/bios/train.pkl', 'rb') as file:
    dicts = pickle.load(file)
uk_jobs_train = pd.DataFrame(dicts)
uk_jobs_train['Id'] = uk_jobs_train.index
uk_jobs_train = uk_jobs_train.reset_index(drop=True)
#we have to replace newlines, otherwise anserini fails
uk_jobs_train['description'] = uk_jobs_train['description'].str.replace('\n',' ')
uk_jobs_train['description'] = uk_jobs_train['description'].str.strip()
uk_jobs_train[['Id','description']].to_csv('/mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/data/bios/train_queries.tsv', sep="\t", index=False,header=False)


with open( '/mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/data/bios/dev.pkl', 'rb') as file:
    dicts = pickle.load(file)
uk_jobs_dev = pd.DataFrame(dicts)
uk_jobs_dev['Id'] = uk_jobs_dev.index
uk_jobs_dev = uk_jobs_dev.reset_index(drop=True)
#we have to replace newlines, otherwise anserini fails
uk_jobs_dev['description'] = uk_jobs_dev['description'].str.replace('\n',' ')
uk_jobs_dev['description'] = uk_jobs_dev['description'].str.strip()
uk_jobs_dev[['Id','description']].to_csv('/mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/data/bios/dev_queries.tsv', sep="\t", index=False,header=False)


with open( '/mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/data/bios/test.pkl', 'rb') as file:
    dicts = pickle.load(file)
uk_jobs_test = pd.DataFrame(dicts)
uk_jobs_test['Id'] = uk_jobs_test.index
uk_jobs_test = uk_jobs_test.reset_index(drop=True)
#we have to replace newlines, otherwise anserini fails
uk_jobs_test['description'] = uk_jobs_test['description'].str.replace('\n',' ')
uk_jobs_test['description'] = uk_jobs_test['description'].str.strip()
uk_jobs_test[['Id','description']].to_csv('/mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/data/bios/test_queries.tsv', sep="\t", index=False,header=False)

#save bm25 output with different schema
train_hits = pd.read_csv('/mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/data/bios/run.bios.bm25.top100.train.tsv', sep=' ', header=None, names=['query_id', 'Q0', 'doc_id', 'rank', 'score','Anserini'])
train_hits = train_hits.drop(['Q0','score','Anserini'], axis='columns')
train_hits.to_csv('/mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/data/bios/bm25.top100.train.tsv', sep="\t", index=False,header=False)

dev_hits = pd.read_csv('/mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/data/bios/run.bios.bm25.top100.dev.tsv', sep=' ', header=None, names=['query_id', 'Q0', 'doc_id', 'rank', 'score','Anserini'])
dev_hits = dev_hits.drop(['Q0','score','Anserini'], axis='columns')
dev_hits.to_csv('/mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/data/bios/bm25.top100.dev.tsv', sep="\t", index=False,header=False)

#generate qrel files:
def create_qrels(data = "train"):
    df_doc = pd.read_csv('/mnt/c/users/cornelia/documents/AI/MasterThesis/IRDebias/data/bios/new_bios.tsv', sep="\t")
    df_doc = df_doc.rename({'Unnamed: 0': 'Id'}, axis=1)
    qrels = {}
    with open(f'/mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/data/bios/{data}.pkl', 'rb') as file:
        dicts = pickle.load(file)
    uk_jobs_train = pd.DataFrame(dicts)
    uk_jobs_train['Id'] = uk_jobs_train.index
    for idx, id in uk_jobs_train['Id'].items():
        job = uk_jobs_train['title'][id]
        #get relevant ids
        qrels[id] = list(df_doc.loc[(df_doc.raw_title == job)]['Id'])

    res = pd.DataFrame(qrels).unstack().droplevel(1).reset_index().rename( columns={'index' :'qid', 0 : 'cid'})
    res['col1'] = 0 
    res['rel'] = 1
    res[['qid', 'col1', 'cid','rel']].to_csv(f'/mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/data/bios/qrels.{data}.tsv', sep="\t", index=False,header=False)

create_qrels("train")
create_qrels("dev")
create_qrels("test")


