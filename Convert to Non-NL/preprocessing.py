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

# data = give data

def remove_paren(test_str):
    ret = ''
    skip1c = 0
    skip2c = 0
    for i in test_str:
        if i == '[':
            skip1c += 1
        elif i == '(':
            skip2c += 1
        elif i == ']' and skip1c > 0:
            skip1c -= 1
        elif i == ')'and skip2c > 0:
            skip2c -= 1
        elif skip1c == 0 and skip2c == 0:
            ret += i
    return ret


def preprocess(sen):
    sen = sen.replace("'''","")
    sen = sen.replace("''","")
    sen = sen.replace('=','')
    sen = sen.replace('References', "")
    sen = remove_paren(sen)
    return sen

text, label = preprocessing(dx)
