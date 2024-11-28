
import nltk


from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# analyzer = SentimentIntensityAnalyzer()
# review = "I love this"
# scores = analyzer.polarity_scores(review)
# print(scores)


from transformers import pipeline
from tqdm import tqdm
import os


#get document collection data
doc_file = '/home/connynic/collection.tsv' #'../GithubRepos/coder/Experiments/data/collection.tsv'
out_file = '/home/connynic/sentimentsVader.tsv'
total_size = sum(1 for _ in open(doc_file))  # simply to get number of lines
token_counts = []
field_sep='\t'
sentiment_pipeline = SentimentIntensityAnalyzer()

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
            res = sentiment_pipeline.polarity_scores(doctext)
            _neg = res['neg']
            _pos = res['pos']
            _neu = res['neu']
            _compound = res['compound']
            fw.write("%s\t%f\t%f\t%f\t%f\n" % (docid, _neg, _pos, _neu, _compound))