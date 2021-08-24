import numpy as np
import pandas as pd
import gdown
from scipy import spatial
import spacy
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words("english")
from tqdm import tqdm
import time
import os
import json


cats = ["Academic degree","Alumnus","Amateur radio","Anthropology","Archaeology","Architecture","Astronomy","Atmosphere","Bioinformatics","Biology"
,"Biotechnology"
,"Botany"
,"Chemistry"
,"Climate"
,"Ecology"
,"Economics"
,"Electronics"
,"Engineering"
,"Genealogy"
,"Geography"
,"Geology"
,"Hydrology"
,"Language",
"Literature"
,"Linguistics"
,"Mathematics"
,"Mechanics"
,"Meteorology"
,"Nuclear"
,"Oceanography"
,"Optics"
,"Physics"
,"Psychology"
,"Research"
,"Robotics"
,"Society"
,"Sociology"
,"Statistics"
,"Zoology"
,"Business",
"Accounting"
,"Advertising"
,"Advisory"
,"Computing"
   ,"Artificial intelligence"
    , "Computer security"
     ,"Database"
     ,"Filename extension"
     ,"Gaming"
    ,"General Computing"
    ,"Hardware"
   ,"Computer network"
   ,"Software"
,   "Technology"
   ,  "Telecom"
   ,"Medicine"
    ,"Autism"
     ,"British Medical Association"
     ,"Cancer"
     ,"Cardiology"
     ,"Dentistry"
      ,"Disability"
     ,"Disease"
     ,"Drug"
     ,"Health care"
     ,"Hospital"
     ,"Human genome"
     ,"Laboratory"
     ,"Medical physics"
     ,"Neurology"
     ,"Nursing"
     ,"Oncology"
     ,"Optometry"
    ,"Orthopedic surgery"
     ,"Pediatrics"
    ,"Pharmacy"
     ,"Physiology"
    ,"Psychiatry"
    ,"Radiology"
    ,"Audit"
  ,  "Bank"
,"Consultant"
 ,   "Finance"
  ,  "Fund"
   , " General Business"
    , "Insurance"
    ,"International business"
    , "Investment"
     ,"Logistics"
     ,"London Stock Exchange"
     ,"Management"
    ,"Marketing"
     ,"Professional association"
    ,"Stock exchange"
     ,"Tax"
    ,"Trade association"
    ,"Community"
    ,"Art",
    "Islam",
    "Christianity"
    ,"Judaism"
    ,"Association"
    ,"Committee"
    ,"Conference"
   ,"Culture"
    , "Development"
    ,"Education"
    ,"Forestry"
    , "History"
    , "Housing"
    ,"Amenity"
    , "Leadership"
    ,"Museum"
    ,"News"
    ,"Media"
     ,"Nonprofit organization"
   ,"Religion"
    , "School"
     ,"Travel"
    ,"Tourism"
    ,"Government"
    ,"Air force"
     ,"Alliance"
    ,"Authority"
     ,"Bureau"
     ,"Council"
     ,"Energy"
    ,"Environment"
     ,"Food and Drug Administration"
    ,"Institute"
    ,"Law"
    ,"Legislation"
     ,"Military"
     ,"NASA"
    ,"Navy"
     ,"Police"
     ,"Politics"
    ,"State"
        ,"Local"
     ,"Transport"
    ,"United Nations",
    "Aircraft"
    ,"Aviation"
    ,"Animal"
     ,"Award"
    ,"Medal"
    ,"Farm"
    ,"Agriculture"
    ,"Food"
    ,"Nutrition"
    ,"Foundation"
    ,"Journal"
    ,"Photography"
    ,"Imaging"
     ,"Science fiction"
     ,"Shipping line"
    ,"Sailing"
    ,"Rehabilitation psychology"
     ,"Surgery"
     ,"Syndrome"
     ,"Therapy"
     ,"Organ transplantation"
     ,"Veterinary medicine"
]



import wikipediaapi

def print_categorymembers(categorymembers, level=0, max_level=1):
        for c in categorymembers.values():
            # print("%s: %s (ns: %d)" % ("*" * (level + 1), c.title, c.ns))
            if 'Category' not in c.title:
                label.append(c.title)
                label_cats.append(i)
            if c.ns == wikipediaapi.Namespace.CATEGORY and level < max_level:
                print_categorymembers(c.categorymembers, level=level + 1, max_level=max_level)

print("Extracting Topics Names")
wiki_wiki = wikipediaapi.Wikipedia('simple')
label = []
label_cats = []
for i in tqdm(cats):
    cat = wiki_wiki.page("Category:" + str(i))
    print_categorymembers(cat.categorymembers)

label = list(set(label))

print("Number of Topics: ",len(label))

label = [x.replace(":","_") for x in label]

wiki_small = wikipediaapi.Wikipedia('simple')
wiki_en = wikipediaapi.Wikipedia('en')

print("Extracting Article Text")
data = []
for ind, title in enumerate(tqdm(label)):
    try:
        page_simple = wiki_wiki.page(title)
        page_en = wiki_en.page(title)
        if(len(page_simple.summary) <10 or len(page_en.summary) < 10):
            continue
        x = dict()
        x['sm_definition'] = page_simple.summary
        # x['en_article'] = " ".join(page_en.text.split())
        x['en_article'] = page_en.text
        x['category'] = label_cats[ind]
        x['title'] = title
        data.append(x)
        if(ind%500 ==0):
            np.save(os.getcwd() + '/auto_pattern_raw.npy', data, allow_pickle=True)
    except:
        time.sleep(5)
        page_simple = wiki_wiki.page(title)
        page_en = wiki_en.page(title)
        if(len(page_simple.summary) <10 or len(page_en.summary) < 10):
            continue
        x = dict()
        x['sm_definition'] = page_simple.summary
        # x['en_article'] = " ".join(page_en.text.split())
        x['en_article'] = page_en.text
        x['category'] = label_cats[ind]
        x['title'] = title
        data.append(x)
        if(ind%500 ==0):
            np.save(os.getcwd() + '/auto_pattern_raw.npy', data, allow_pickle=True)

print("Saving data")
np.save(os.getcwd() + '/auto_pattern_raw.npy', data, allow_pickle=True)
with open(os.getcwd() + '/auto_pattern_raw.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)
