from transformers import pipeline
from tqdm import tqdm
import os


#get document collection data
doc_file = '/home/connynic/collection.splits0.tsv' #'../GithubRepos/coder/Experiments/data/collection.tsv'
out_file = '/home/connynic/sentimentshuggingface.splits0.tsv'
total_size = sum(1 for _ in open(doc_file))  # simply to get number of lines
token_counts = []
field_sep='\t'
sentiment_pipeline = pipeline("sentiment-analysis")

labels = []
scores = []

with open(out_file, "w", encoding="utf8") as fw:
    with open(doc_file, "r", encoding="utf8") as fr:
        for line in tqdm(fr, total=total_size):
            vals = line.strip().split('\t')
            if len(vals) != 2:
                print("Failed parsing the line (skipped):\n %s " % line.strip())
                continue
            docid = vals[0]
            doctext = vals[1]
            res = sentiment_pipeline(doctext)[0]
            _sentiment = res['label']
            _score = res['score']
            fw.write("%s\t%s\t%f\n" % (docid, _sentiment, _score))



# for line in tqdm(open(doc_file), total=total_size, desc=f"Tokenize: {os.path.basename(doc_file)}"):
#     seq_id, *text = line.split(field_sep)  # accommodates the case of title or other fields
#     text = (" ").join(text)  # in case of more than 1 fields, joins with " [SEP] "
#     res = sentiment_pipeline(text)[0]
#     print(res)
#     labels.append(res['label'])
#     scores.append(res['score'])
#     break
#     #outFile.write(json.dumps({"id": seq_id, "ids": ids}))
#     #outFile.write("\n")

# print(scores)

# data = ["I love you", "I hate you"]
# res = sentiment_pipeline(data)
# print(res)


