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

data1 = np.load(os.getcwd() + '/auto_data_v2.npy', allow_pickle = True) # path to auto pattern v2

data2 = np.load(os.getcwd() + '/auto_data_v3.npy', allow_pickle = True) # path to auto pattern v3

data = np.concatenate([data1, data2])
random.seed(0)
random.shuffle(data)

def process_data(data):
    train = pd.DataFrame(columns=["text", 'summary'])
    val = pd.DataFrame(columns=["text", 'summary'])
    train_data = data[:int(len(data)*0.95)]
    val_data = data[int(len(data)*0.95):]
    for i, dic in enumerate(tqdm(train_data)):
        train.loc[i] = [dic['en_article']] + [dic['sm_definition']]
    for i, dic in enumerate(tqdm(val_data)):
        val.loc[i] = [dic['en_article']] + [dic['sm_definition']]
    return train, val

print("preparing data")
train, val = process_data(data)


train['text'].replace('', np.nan, inplace=True)
train['summary'].replace('', np.nan, inplace=True)
val['text'].replace('', np.nan, inplace=True)
val['summary'].replace('', np.nan, inplace=True)
train = train.dropna()
val = val.dropna()

train.to_csv(os.getcwd() + '/auto_pattern_v4/train_v4.csv', index=False)
val.to_csv(os.getcwd() + '/auto_pattern_v4/val_v4.csv', index=False)