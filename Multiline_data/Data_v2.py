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

data = np.load('path to auto data(Datasets/auto_data_v2)', allow_pickle= True)
df = copy.deepcopy(data)

# break sentence
def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.label() =='NP'):
        yield subtree.leaves()
        
def get_word_postag(word):
    if pos_tag([word])[0][1].startswith('J'):
        return wordnet.ADJ
    if pos_tag([word])[0][1].startswith('V'):
        return wordnet.VERB
    if pos_tag([word])[0][1].startswith('N'):
        return wordnet.NOUN
    else:
        return wordnet.NOUN

def get_terms(tree):    
    for leaf in leaves(tree):
        terms = [w for w,t in leaf]
        yield terms

# Rule for NP chunk and VB Chunk
grammar = r"""
    NBAR:
        {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
        {<RB.?>*<VB.?>*<JJ>*<VB.?>+<VB>?} # Verbs and Verb Phrases
        
    NP:
        {<NBAR>}
        {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
        
"""


#word tokenizeing and part-of-speech tagger

def split_syn(corpus):
    sent = splitter(corpus)
    sent = [str(x) for x in sent.sents]
    mod_sent = []
    for document in sent:
        tokens = [nltk.word_tokenize(sent) for sent in [document]]
        postag = [nltk.pos_tag(sent) for sent in tokens][0]


        #Chunking
        cp = nltk.RegexpParser(grammar)

        # the result is a tree
        tree = cp.parse(postag)


        terms = get_terms(tree)

        features = []
        for term in terms:
            _term = ''
            for word in term:
                _term += ' ' + word
            features.append(_term.strip())
        if(len(features)==0):
            continue
        mod_sent.append(". ".join(features) + '.')
        # print(". ".join(features))
        # print(document)
        # print(" ")
    return " ".join(mod_sent)
    

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


####### data creation

sm_defin  = [x['sm_definition'] for x in data]

def process_data(data):
    dataset = copy.deepcopy(data)
    for ind, text in enumerate(tqdm(dataset)):
        x = dataset[ind]['en_article'].split('\n\nSee also')[0]
        x = x.split('References')[0]
        x = preprocess(x)
        extra = random.sample(list(sm_defin), 2)
        shuffle(extra)
        x = x[:7000]
        extra.append(x)
        # sections = x.split('\n\n')
        # for i,j in enumerate(sections):
        #     joined = j.replace('\n', '. ')
        #     sections[i] = joined 
        # shuffle(sections)
        final_article = ''
        for i,j in enumerate(extra):
            final_article += j + '. '
        final_article = final_article.replace('..','.')
        final_article = final_article.replace('\n',' ')
        final_article = final_article.replace('References','')
        final_article = final_article.replace('==','')
        final_article = " ".join(final_article.split())
        final_article = split_syn(final_article)
        dataset[ind]['en_article'] = final_article
    return dataset

x = process_data(data)