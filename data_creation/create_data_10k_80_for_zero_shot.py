import json
import logging
import random
import string
import sys
import numpy as np
import pre_processing_utils as proc
import re
# for entity-expansion, change number_of_entity_trials

def main():
    random.seed(1012)
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger = logging.getLogger(__name__)
    chars = string.ascii_lowercase
    number_of_entity_trials = 1
    fictitious_entities = proc.generate_pairs_of_random_strings(number_of_pairs=number_of_entity_trials, 
                                                                min_length=3,
                                                                max_length=12,
                                                                character_set=chars)
    mask_prediction_options = []

    # test_byset = []
    # for i in range(170):
    #     test_byset.append(str(i))
    # eval_byset = []
    # for j in range(170, 340, 1):
    #     eval_byset.append(str(j))
    # train_byset = []
    # for k in range(340, 1190, 1):
    #     train_byset.append(str(k))
    eval_set = []
    test_set = []
    for i in range(767):
        eval_set.append(str(i))
    for j in range(767, 1706, 1):
        test_set.append(str(j))
    statements = []
    with open("../data/truism_data/RICA_10k_probe_sets.json", "r") as f:
        data = json.load(f)
    #     probes_set = {}
        perturbations = ['original', 'negation']
        asymetric = ['original', 'asymmetric_premise', 'asymmetric_conclusion']
    #     right_probes = []
    #     wrong_probes = []
        


        test_statements = []
        train_statements = []
        eval_statements = []
        for i in data:
            for pert in perturbations:
                random.seed()
                fictitious_entities = proc.generate_pairs_of_random_strings(number_of_pairs=number_of_entity_trials, 
                                                                min_length=3,
                                                                max_length=12,
                                                                character_set=chars)
                for order in asymetric:
                    if len(data[i][pert][order]) != 0:
                        # random.seed()
                        # fictitious_entities = proc.generate_pairs_of_random_strings(number_of_pairs=number_of_entity_trials, 
                        #                                         min_length=3,
                        #                                         max_length=12,
                        #                                         character_set=chars)
                        statement = data[i][pert][order][0]
                        rignt_answer = data[i][pert][order][2]
                        wrong_answer = data[i][pert][order][3]
                        
                        for entity_pair in fictitious_entities:
                            new_statement = re.sub(r"\bA\b", entity_pair[0], statement)
                            new_statement = re.sub(r"\bB\b", entity_pair[1], new_statement)
                            # statements.append((new_statement, rignt_answer, [rignt_answer, wrong_answer]))
                            if i in eval_set:
                                eval_statements.append((new_statement, rignt_answer, [rignt_answer, wrong_answer]))
                            elif i in test_set:
                                test_statements.append((new_statement, rignt_answer, [rignt_answer, wrong_answer]))
                            # if i not in test_byset and i not in eval_byset and i in train_byset:
                            #     train_statements.append((new_statement, rignt_answer, [rignt_answer, wrong_answer]))
                            # elif i in test_byset:
                            #     test_statements.append((new_statement, rignt_answer, [rignt_answer, wrong_answer]))
                            # elif i in eval_byset:
                            #     eval_statements.append((new_statement, rignt_answer, [rignt_answer, wrong_answer]))
    
    with open('../data/100k/test_sentences.txt','w') as train:
        for i, sent in enumerate(test_statements):
            train.write(sent[0])
            if i < len(test_statements) - 1:
                train.write('\n')
    with open('../data/100k/test_sentences_m.txt','w') as train_m:
        for i, sent in enumerate(test_statements):
            train_m.write(sent[1])
            if i < len(test_statements) - 1:
                train_m.write('\n')
    with open('../data/100k/config.txt','w') as config:
        for i, sent in enumerate(test_statements):
            config.write(str(sent[2]))
            if i < len(test_statements) - 1:
                config.write('\n')
    
    with open('../data/100k/eval_sentences.txt','w') as train:
        for i, sent in enumerate(eval_statements):
            train.write(sent[0])
            if i < len(eval_statements) - 1:
                train.write('\n')
    with open('../data/100k/eval_sentences_m.txt','w') as train_m:
        for i, sent in enumerate(eval_statements):
            train_m.write(sent[1])
            if i < len(eval_statements) - 1:
                train_m.write('\n')

    # with open('../data/10k_byset_random/eval_sentences.txt','w') as train:
    #     for i, sent in enumerate(eval_statements):
    #         train.write(sent[0])
    #         if i < len(eval_statements) - 1:
    #             train.write('\n')
    # with open('../data/10k_byset_random/eval_sentences_m.txt','w') as train_m:
    #     for i, sent in enumerate(eval_statements):
    #         train_m.write(sent[1])
    #         if i < len(eval_statements) - 1:
    #             train_m.write('\n')
    # with open('../data/10k_byset_random/test_sentences.txt','w') as train:
    #     for i, sent in enumerate(test_statements):
    #         train.write(sent[0])
    #         if i < len(test_statements) - 1:
    #             train.write('\n')
    # with open('../data/10k_byset_random/test_sentences_m.txt','w') as train_m:
    #     for i, sent in enumerate(test_statements):
    #         train_m.write(sent[1])
    #         if i < len(test_statements) - 1:
    #             train_m.write('\n')
    # with open('../data/10k_byset_random/config.txt','w') as config:
    #     for i, sent in enumerate(test_statements):
    #         config.write(str(sent[2]))
    #         if i < len(test_statements) - 1:
    #             config.write('\n')
    




if __name__ == "__main__":
    main()



