import os
import numpy as np
from pathlib import Path
home = str(Path.home())
from urllib import request
from bs4 import BeautifulSoup

import nltk
from nltk import pos_tag
from nltk.tag.stanford import StanfordNERTagger
from nltk.tokenize import word_tokenize
from nltk.chunk import conlltags2tree
from nltk.tree import Tree
from nltk.internals import find_jars_within_path

# Process text  
def process_text(text):
	token_text = word_tokenize(text)
	return token_text

# Stanford NER tagger    
def stanford_tagger(token_text):
    classifier_path = home+'/nltk_data/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz'
    stanford_path = home+'/nltk_data/stanford-ner/stanford-ner.jar'
    st = nltk.tag.stanford.CoreNLPNERTagger(url='http://localhost:9000')
    ne_tagged = st.tag(token_text)
    return(ne_tagged)
 
# NLTK POS and NER taggers   
def nltk_tagger(token_text):
    tagged_words = nltk.pos_tag(token_text)
    ne_tagged = nltk.ne_chunk(tagged_words)
    return(ne_tagged)

# Tag tokens with standard NLP BIO tags
def bio_tagger(ne_tagged):
    bio_tagged = []
    prev_tag = "O"
    for token, tag in ne_tagged:
        if tag == "O": #O
            bio_tagged.append((token, tag))
            prev_tag = tag
            continue
        if tag != "O" and prev_tag == "O": # Begin NE
            bio_tagged.append((token, "B-"+tag))
            prev_tag = tag
        elif prev_tag != "O" and prev_tag == tag: # Inside NE
            bio_tagged.append((token, "I-"+tag))
            prev_tag = tag
        elif prev_tag != "O" and prev_tag != tag: # Adjacent NE
            bio_tagged.append((token, "B-"+tag))
            prev_tag = tag
    return bio_tagged

# Create tree       
def stanford_tree(bio_tagged):
    tokens, ne_tags = zip(*bio_tagged)
    pos_tags = [pos for token, pos in pos_tag(tokens)]

    conlltags = [(token, pos, ne) for token, pos, ne in zip(tokens, pos_tags, ne_tags)]
    ne_tree = conlltags2tree(conlltags)
    return ne_tree

def structure_locations(ne_tree):
    locations = []
    for subtree in ne_tree:
        if type(subtree) == Tree: # If subtree is a noun chunk, i.e. NE != "O"
            if subtree.label() == "LOCATION":
                locations.append(" ".join([token for token, pos in subtree.leaves()]))
    return locations

def stanford_extract_locations(text):
    stanford_tree_structure = stanford_tree(bio_tagger(stanford_tagger(process_text(text))))
    locations = structure_locations(stanford_tree_structure)
    return locations

def structure_entities(ne_tree):
    people = []
    companies = []
    locations = []
    for subtree in ne_tree:
        if type(subtree) == Tree: # If subtree is a noun chunk, i.e. NE != "O"
            if subtree.label() == "PERSON":
                people.append(" ".join([token for token, pos in subtree.leaves()]))
            elif subtree.label() == "ORGANIZATION":
                companies.append(" ".join([token for token, pos in subtree.leaves()]))
            elif subtree.label() == "LOCATION":
                locations.append(" ".join([token for token, pos in subtree.leaves()]))
    return people,companies,locations

def stanford_extract_entities(text):
    stanford_tree_structure = stanford_tree(bio_tagger(stanford_tagger(process_text(text))))
    return structure_entities(stanford_tree_structure)

if __name__ == '__main__':
    url = "http://www.capp.ca/about-us/our-organization/executive-team"

    html = request.urlopen(url).read().decode('utf8')
    soup = BeautifulSoup(html,'html.parser')
    raw = soup.get_text(" ",strip=True)
    tags = soup.find_all(['h1','h2','h3','h4','h5','h6','p','a'])
    texts = list(filter(None,[tag.get_text(" ", strip=True) for tag in tags]))

    print("** NLTK Stanford ")
    people = []
    companies = []
    for text in texts:
        local_people, local_companies = stanford_extract_entities(text)
        people = people + local_people
        companies = companies + local_companies
        
    people = list(set(people))
    companies = list(set(companies))
    print("### People : ###\n"+str(people))
    print("### Companies : ###\n"+str(companies))