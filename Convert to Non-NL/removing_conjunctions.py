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
from random import shuffle

# test -> list of articles to be broken

tool = spacy.load("en_core_web_sm")
mod_text = []
mod_labels = []
for k,m in enumerate(tqdm(text)):
    sent = tool(m)
    sent = [str(x) for x in sent.sents]
    mod_sent = []
    for sen in sent:
        tags = tool(sen)
        words = []
        for token in tags:
            if token.tag_ == 'CC':
                if len(words)>0 and last_token!= ',':
                    mod_sent.append(" ".join(words) + '.')
                    words = []
            else:
                words.append(str(token))
            last_token = str(token)
        if len(words)!=0:
            # print(words)
            mod_sent.append(" ".join(words))
#     shuffle(mod_sent)
    para = " ".join(mod_sent)
    para = " ".join(para.split())
    mod_text.append(" ".join(mod_sent))

data = []
for i, j in enumerate(mod_text):
    x = dict()
    x['definition'] = text[i]
    x['mod_definition'] = mod_text[i]
    data.append(x)


