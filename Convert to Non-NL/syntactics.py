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

text = '''
some_text
'''
result = split_syn(text)
    