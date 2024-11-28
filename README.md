# DebiasingConGater

This repository is dedicated to the models of the paper [Controllable Gate Adapters (ConGater)](https://arxiv.org/abs/2401.16457) Accepted to main conference of EACL 2024.

The infrastructure of the code is similar to the [ModularizedDebiasing](https://github.com/CPJKU/ModularizedDebiasing) from the papers:

[Modular and On-demand Bias Mitigation with Attribute-Removal Subnetworks](https://aclanthology.org/2023.findings-acl.386.pdf)

and 

[Parameter-efficient Modularised Bias Mitigation via AdapterFusion](https://aclanthology.org/2023.eacl-main.201.pdf)


# Tasks and Environments

## Classification Task 

### Dataset 
- All the datasets for the classification task are publicly available for download. 
- We have a pre-processed version of the data for train, eval and test in seperate .pkl file which we use to call the data loader and Dataset file
- Due to the size of the dataset we are not able to provide them in the repository but we provided the config file for the dataset (Without Path) 

### Environment 
To Install the environment after instilling Conda or Mini conda run the command 

```
conda env create -f cls_congater.yml
```

After installation is complete you can access the environment by running :

```
conda activate cls_congater
```

### Structure of the code 

* **scripts**: Folder contains utility codes for adversarial mode, checkpoint, generating embeddings and etc. 

* **src**: Folder contains utility codes for attacking , evaluating , logging and data handler

* **models**: Subfolder of src contains all the models that can be called for training which contain , Baseline, Baseline adv, Adapter, Adapter Adv and ConGater 

* **cfg.yml**: contains all the training and dataset configurations required to run the training with default values

* **main_attack.py**: can be used to attack an already trained model (normally called with main.py)

* **main.py**: contains train wrappers and arguments to overwrite default config file. 

### Changing Code using Arguments:

To run the code you can use the following python command.

```
python main.py --gpu_id=0 --ds=hatespeech --model_type=congater --training_method=par --model_name=mini --random_seed --num_runs=1 --gate_squeeze_ratio=12 --log_wandb
```

* **--gpu_id**: sets the id of the gput that you are using by default 0 
* **--ds**: sets the dataset that you want to run the model on 
* **--model_type**: sets the model type that you want to run (baseline, baseline_adv,adapter_baseline, adapter_adv, congater)
* **--training_method**: For ConGater you can select parallel (par) training or post-hoc(post) training 
* **--model_name**: sets the model to run (bert,mini,roberta-base) are the values 
* **--random_seed**: activates random seeding for several runs to insure we dont have seed selection bias
* **--num_runs**: select how many times you want the current cofig to run , (always use with --random_seed)
* **--gate_squeeze_ratio**: selects the bottleneck for the ConGater , for adapter the default values are in the cofig based on previous papers
* **--log_wandb**: Starts logging the runs on wandb 
* **--wandb_project**: sets the prefix of the project, dataset name is added to the project name by default

For More control arguments please check out main.py file line 543-580.


# Paper citation:
```
@inproceedings{masoudian2024congater,
	title        = {Effective Controllable Bias Mitigation for Classification and Retrieval using Gate Adapters},
	author       = {Shahed Masoudian and Cornelia Volaucnik and Markus Schedl and Navid Rekabsaz},
	year         = 2024,
	booktitle    = {Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics},
	publisher    = {Association for Computational Linguistics},
	address      = {Malta}
}
```


# coderdebiasinf




## Add your files
Code by : Cornelia Volaucnik

```
## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.


## Installation

Install conda and then run 

```bash
conda env create -f environment.yml
```

Downgrade pytorch by applying:
```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install pandas==1.2.3
pip install optuna==2.5.0
pip install ipdb==0.13.5
pip install xlrd==2.0.1
pip install xlwt==1.3.0
pip install xlutils==2.0.0
pip install psutil==5.8.0
pip install adapter-transformers
```

and then activate the environment.

## Prepare bios data for model

Create the anserini candidates by running these statements:

With this statement the collection data is converted into jsonl format (code borrowed from msmarco data wrangling):
```bash
python tools/scripts/msmarco/convert_collection_to_jsonl.py   --collection-path /mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/data/bios/documents.tsv   --output-folder collections/bios/collection_jsonl
```

To generate the index:

```bash
target/appassembler/bin/IndexCollection \
  -collection JsonCollection \
  -input collections/bios/collection_jsonl \
  -index indexes/bios/lucene-index-bios \
  -generator DefaultLuceneDocumentGenerator \
  -threads 9 -storePositions -storeDocvectors -storeRaw 
```

To generate the candidates, save the query files with the document_prep.py file and then:

```bash
target/appassembler/bin/SearchCollection   -index indexes/bios/lucene-index-bios   -topics /mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/data/bios/train_queries.tsv   -topicreader TsvInt   -output runs/run.bios.bm25.top100.train.tsv.tsv   -parallelism 4   -bm25 -bm25.k1 0.82 -bm25.b 0.68 -hits 100
```

And for dev:
```bash
target/appassembler/bin/SearchCollection   -index indexes/bios/lucene-index-bios   -topics /mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/data/bios/dev_queries.tsv   -topicreader TsvInt   -output runs/run.bios.bm25.top100.dev.tsv.tsv   -parallelism 4   -bm25 -bm25.k1 0.82 -bm25.b 0.68 -hits 100
```

For test:
```bash
target/appassembler/bin/SearchCollection   -index indexes/bios/lucene-index-bios   -topics /mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/data/bios/test_queries.tsv   -topicreader TsvInt   -output runs/run.bios.bm25.top100.test.tsv   -parallelism 4   -bm25 -bm25.k1 0.82 -bm25.b 0.68 -hits 100
```

Create the tokenized files with the bi-encoder tokenizer:

```bash
python /mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/GithubRepos/coder/coder/convert_text_to_tokenized.py --output_dir . --collection documents.tsv --queries 'train_queries.tsv' --tokenizer_from "/mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/coderdebiasinf/results/bi-encoder/bi-encoder-distilbert-base-uncased-2023-08-10_15-51-20-latest"
```

For dev:

```bash
python /mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/GithubRepos/coder/coder/convert_text_to_tokenized.py --output_dir . --collection documents.tsv --queries 'dev_queries.tsv' --tokenizer_from "/mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/coderdebiasinf/results/bi-encoder/bi-encoder-distilbert-base-uncased-2023-08-10_15-51-20-latest"
```

For test:

```bash
python /mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/GithubRepos/coder/coder/convert_text_to_tokenized.py --output_dir . --collection documents.tsv --queries 'test_queries.tsv' --tokenizer_from "/mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/coderdebiasinf/results/bi-encoder/bi-encoder-distilbert-base-uncased-2023-08-10_15-51-20-latest"
```

Create memmaps:
```bash
python /mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/GithubRepos/coder/coder/create_memmaps.py --tokenized_collection documents.tokenized.json --output_collection_dir collection_memmap --max_doc_length 512
```

### Create embedding memmap
```bash
python /mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/GithubRepos/coder/coder/precompute.py --model_type huggingface --encoder_from "/mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/coderdebiasinf/results/bi-encoder/bi-encoder-distilbert-base-uncased-2023-08-10_15-51-20-latest" --collection_memmap_dir  collection_memmap/ --output_dir . --max_doc_length 512
```

### Create memmaps for queries
The number of max_candidates is important here, otherwise Georges code fails (this is a small bug)
```bash
python /mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/GithubRepos/coder/coder/create_memmaps.py --candidates bm25.top100.train.tsv --output_candidates_dir bm25.top100.train_memmap --max_candidates 100
```

```bash
python /mnt/c/Users/Cornelia/Documents/AI/MasterThesis/IRDebias/GithubRepos/coder/coder/create_memmaps.py --candidates bm25.top100.dev.tsv --output_candidates_dir bm25.top100.dev_memmap --max_candidates 100
```

To create the document neutrality we could try something like this:

```bash
python calc_documents_neutrality.py --collection-path /mnt/c/Users/Cornelia/Documents/AI/Masterthesis/IRDebias/data/bios/documents.tsv --representative-words-path ./resources/wordlist_protectedattribute_gender.txt --threshold 1 --out-file /mnt/c/Users/Cornelia/Documents/AI/Masterthesis/IRDebia
s/data/bios/collection_neutralityscores.tsv --groups-portion """{\"f\":1, \"m\":0}"""
```





