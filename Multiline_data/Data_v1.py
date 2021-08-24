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
import time
import os
import json
from random import shuffle
import copy
from tqdm.notebook import tqdm
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from IPython.display import display
splitter = spacy.load("en_core_web_sm")
import random

#data for articles

auto_data = []
for ind, article in enumerate(tqdm(data)):
    x = random.sample(list(data), 3)
    x.append(article)
    shuffle(x)
    mod_article = dict()
    mod_article['definition'] = ''
    for i,j in enumerate(x):
        mod_article['definition'] += j['definition'][j['definition'].index(".") + 2:] +'. '
    mod_article['definition'] = 'What is ' + article['topic'] + '.\t' + mod_article['definition'] 
    mod_article['topic'] = article['topic']
    mod_article['summary'] = article['summary']
    auto_data.append(mod_article)

np.save('PATH', auto_data)
