import re
import sys

# Credit: reuse of code from HW4, which is a re-implementation of the paper "AnthroScore: A Computational Linguistic Measure of Anthropomorphism"
import pandas as pd
import argparse
import spacy
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM
import numpy as np
import scipy
import gc
import seaborn as sns 
from matplotlib import pyplot as plt

nlp = spacy.load("en_core_web_sm")


def get_human_nonhuman_scores(sentence, human, nonhuman, model, tokenizer, device):
    human_inds = [tokenizer.get_vocab()[x] for x in human]
    nonhuman_inds = [tokenizer.get_vocab()[x] for x in nonhuman]
    
    ########################################
    ########### PART 1 #####################
    ########################################

    # sometimes sentence may get too long for the BERT model, so we need to truncate
    inputs = tokenizer(sentence, truncate=True, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    masked_token_logits = logits[0, mask_token_index]
    softmax_probability = masked_token_logits.softmax(dim=1)

    human_scores = 0
    nonhuman_scores = 0

    for ind in human_inds:
        human_scores += softmax_probability[0, ind]

    for ind in nonhuman_inds:
        nonhuman_scores += softmax_probability[0, ind]
    
    return human_scores, nonhuman_scores


def get_anthroscore(text, entities, model, tokenizer, device):
    # Mask sentences
    pattern_list = ['\\b%s\\b'%s for s in entities] # add boundaries
    masked_sents = []
    if text.strip():
        doc = nlp(text)
        for _parsed_sentence in doc.sents:
            for _noun_chunk in _parsed_sentence.noun_chunks:
                if _noun_chunk.root.dep_ == 'nsubj' or _noun_chunk.root.dep_ == 'dobj':
                    for _pattern in pattern_list:
                        if re.findall(_pattern.lower(), _noun_chunk.text.lower()):
                                _verb = _noun_chunk.root.head.lemma_.lower()
                                target = str(_parsed_sentence).replace(str(_noun_chunk),'<mask>')
                                masked_sents.append(target)

    if len(masked_sents)==0:
        #print("Stopping calculation, no words found.")
        return np.nan
        
    # Get scores
    hterms = ['he', 'she', 'her', 'him', 'He', 'She', 'Her']
    nterms = ['it', 'its', 'It', 'Its']
    anthroscore = 0
    ########################################
    ########### PART 1 #####################
    ########################################
    for sentence in masked_sents:
        human_score, nonhuman_score = get_human_nonhuman_scores(sentence, hterms, nterms, model, tokenizer, device)
        anthroscore = anthroscore + torch.log(human_score) - torch.log(nonhuman_score)

    anthroscore /= len(masked_sents)

    anthroscore = anthroscore.item()
    
    return anthroscore


