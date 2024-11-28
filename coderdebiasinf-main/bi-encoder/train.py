#Load Data
import argparse

import pickle
import pandas as pd

from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, losses, SentenceTransformer, evaluation

import logging
from datetime import datetime
import torch

from tqdm.autonotebook import tqdm

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu no.")
    parser.add_argument("--seed", type=int, default=0, help="torch random seed")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased", help="model name")
    base_args, _ = parser.parse_known_args()

    model_name = base_args.model 
    #train_batch_size = 64
    num_epochs = 4
    model_save_path = '/mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/coderdebiasinf/results/bi-encoder/bi-encoder-'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    #pos_neg_ration = 4
    torch.manual_seed(base_args.seed)
    print(f"torch.manual_seed({base_args.seed})")

    device = torch.device(f"cuda:{int(base_args.gpu_id)}")

    with open( '/mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/data/bios/new_bios.pkl', 'rb') as file:
        dicts = pickle.load(file)
    bios = pd.DataFrame(dicts).reset_index()
    bios = bios.drop('index', axis=1)


    with open( '/mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/data/bios/train.pkl', 'rb') as file:
        dicts = pickle.load(file)
    uk_jobs_train = pd.DataFrame(dicts)

    with open( '/mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/data/bios/dev.pkl', 'rb') as file:
        dicts = pickle.load(file)
    uk_jobs_dev = pd.DataFrame(dicts)

    dev_hits = pd.read_csv('/mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/data/bios/run.bios.bm25.top100.dev.tsv', sep=' ', header=None, names=['query_id', 'Q0', 'doc_id', 'rank', 'score','Anserini'])
    train_hits = pd.read_csv('/mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/data/bios/run.bios.bm25.top100.train.tsv', sep=' ', header=None, names=['query_id', 'Q0', 'doc_id', 'rank', 'score','Anserini'])
    dev_hits = dev_hits.drop(['Q0','rank','score','Anserini'], axis='columns')
    dev_hits['doc_id'] = dev_hits['doc_id'].replace('doc','',regex=True).astype(int)
    train_hits = train_hits.drop(['Q0','rank','score','Anserini'], axis='columns')
    train_hits['doc_id'] = train_hits['doc_id'].replace('doc','',regex=True).astype(int)

    model = SentenceTransformer(model_name, device=device)

    #In Train and Test for every datapoint with label 1 we add 4 negative samples
    import random
    from sentence_transformers import InputExample
    from tqdm.autonotebook import tqdm

    train_samples = []

    for id in tqdm(range(len(uk_jobs_train))):

        new_bios = bios.loc[train_hits[train_hits['query_id']==id]['doc_id']]

        query = uk_jobs_train['description'][id]
        job = uk_jobs_train['title'][id]
        
        try:
            #pos_passage = random.choice(list(new_bios.loc[(bios.raw_title == job)&(bios.gender == 'M')]['bio']))
            pos_passage = random.choice(list(new_bios.loc[(bios.raw_title == job)]['bio']))
            train_samples.append(InputExample(texts=[query, pos_passage], label=1.0))
            pos_passage = random.choice(list(new_bios.loc[(bios.raw_title == job)&(bios.bio != pos_passage)]['bio']))
            train_samples.append(InputExample(texts=[query, pos_passage], label=1.0))
        except:
            print(f"No positive sample for Id {id}")
        
        for looper in range(4):
            neg_passage = random.choice(list(new_bios.loc[bios.raw_title != job]['bio']))
            train_samples.append(InputExample(texts=[query, neg_passage],label=0.0))


    dev_samples = {}

    for id in tqdm(range(len(uk_jobs_dev))):

        new_bios = bios.loc[dev_hits[dev_hits['query_id']==id]['doc_id']]

        query = uk_jobs_dev['description'][id]
        job = uk_jobs_dev['title'][id]

        dev_samples[id] = {'query': query, 'positive': set(), 'negative': set()}

        pos_passage = random.choice(list(new_bios.loc[bios.raw_title == job]['bio']))

        neg_passage = random.choice(list(new_bios.loc[bios.raw_title != job]['bio']))

        dev_samples[id]['positive'].add(pos_passage)
        dev_samples[id]['negative'].add(neg_passage)


    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)

    # It performs a classification task and measures scores like F1 (finding relevant passages) and Average Precision
    evaluator = evaluation.RerankingEvaluator(dev_samples, name='train-eval')
    train_loss = losses.CosineSimilarityLoss(model)

    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])


    warmup_steps = 500
    logging.info("Warmup-steps: {}".format(warmup_steps))

    torch.cuda.empty_cache()
    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=5904,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          use_amp=True)

    #Save latest model
    model.save(model_save_path+'-latest')

if __name__ == "__main__":

    main()
