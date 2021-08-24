import numpy as np
import pandas as pd
import gdown
from scipy import spatial
import spacy
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words("english")
from tqdm.notebook import tqdm

from sentence_transformers import SentenceTransformer
model = SentenceTransformer(PATH)

### text is inpu data, label is the topic name for data

data = []
for i,j in enumerate(text):
    data.append([text[i],label[i]])


def process(data_point):
    print("working")
    sent = splitter(data_point[0])
    sent = [str(x) for x in sent.sents]
    mod_sent = []
    for cnt,eg in enumerate(sent):
        print(cnt)
        x = model.encode(data_point[1])
        eg = " ".join(eg.split())
        lis = eg.split(" ")
        k=0
        score = np.zeros((len(lis),1))
        for i, j in enumerate(lis):
            y = model.encode(lis[i])
            cs = sim(x, y)
            if i==0:
                if cs<0.15:
                    score[i]+=1
            elif i==len(lis)-1:
                if cs<0.15:
                    score[i]+=1
            else:
                if cs<0.15:
                    score[i]+=1
                if(cs > sim(x, model.encode(str(lis[i-1]) +str(" ") + str(lis[i])))):
                    score[i-1] +=1
                if(cs > sim(x, model.encode( str(lis[i]) + str(" ") +str(lis[i+1]) ))):
                    score[i+1] +=1
        words = []
        for i,j in enumerate(lis):
            if(j.lower() in stop_words):
                score[i]+=2
            if(score[i]>2):
                continue
            words.append(j)
        mod_sent.append(" ".join(words))
        # print(" ".join(words))
    mod_para = " ".join(mod_sent)
    x = dict()
    x['definition'] = data_point[0]
    x['mod_definition'] = mod_para
    print("ending")
    return x

