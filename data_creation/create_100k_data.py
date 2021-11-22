import json
import logging
import random
import string
import sys
import numpy as np
import pre_processing_utils as proc
import re


with open('../data/RICA_100k_probe_sets.json','r') as file:
    data = json.load(file)

    chars = string.ascii_lowercase
    number_of_entity_trials = 5
    statements = []
    for i in data:
        if len(data[i]) != 0:
            right_statement = data[i][0][0]
            wrong_statement = data[i][0][1]
            right_answer = data[i][0][2]
            wrong_answer = data[i][0][3]
            print(right_statement)
            random.seed()
            fictitious_entities = proc.generate_pairs_of_random_strings(number_of_pairs=number_of_entity_trials, 
                                                                min_length=3,
                                                                max_length=12,
                                                                character_set=chars)
            
            for entity_pair in fictitious_entities:
                new_statement = re.sub(r"\bA\b", entity_pair[0], right_statement)
                new_statement = re.sub(r"\bB\b", entity_pair[1], new_statement)
                statements.append((new_statement, right_answer, [right_answer, wrong_answer]))



with open('../data/100k/'+str(number_of_entity_trials)+'/train_sentences.txt','w') as train:
        for i, sent in enumerate(statements):
            train.write(sent[0])
            if i < len(statements) - 1:
                train.write('\n')
with open('../data/100k/'+str(number_of_entity_trials)+'/train_sentences_m.txt','w') as train_m:
    for i, sent in enumerate(statements):
        train_m.write(sent[1])
        if i < len(statements) - 1:
            train_m.write('\n')