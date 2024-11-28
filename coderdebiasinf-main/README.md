# coderdebiasinf




## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.cp.jku.at/cornelia-volaucnik/coderdebiasinf.git
git branch -M main
git push -uf origin main
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

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.


## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
