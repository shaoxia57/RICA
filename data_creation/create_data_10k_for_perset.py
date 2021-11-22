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
    number_of_entity_trials = 10
    fictitious_entities = proc.generate_pairs_of_random_strings(number_of_pairs=number_of_entity_trials, 
                                                                min_length=3,
                                                                max_length=12,
                                                                character_set=chars)
    mask_prediction_options = []

    with open("../data/truism_data/RICA_10k_probe_sets.json", "r") as f:
        probes = json.load(f)

    # statements = proc.prep_ft_instances_for_10k(probes, fictitious_entities,number_of_entity_trials)

    with open("../data/truism_data/RICA_10k_probe_sets.json", "r") as f:
        data = json.load(f)
    #     probes_set = {}
        perturbations = ['original', 'negation']
        asymetric = ['original', 'asymmetric_premise', 'asymmetric_conclusion']
    #     right_probes = []
    #     wrong_probes = []
        train_statements = []
        test_statements = []
        eval_statements = []
        for i in data:
            for pert in perturbations:
                random.seed()
                fictitious_entities = proc.generate_pairs_of_random_strings(number_of_pairs=number_of_entity_trials, 
                                                                min_length=3,
                                                                max_length=12,
                                                                character_set=chars)
                for order in asymetric:
                    # random.seed()
                    # fictitious_entities = proc.generate_pairs_of_random_strings(number_of_pairs=number_of_entity_trials, 
                    #                                             min_length=3,
                    #                                             max_length=12,
                    #                                             character_set=chars)
                    if len(data[i][pert][order]) != 0:
                        statement = data[i][pert][order][0]
                        rignt_answer = data[i][pert][order][2]
                        wrong_answer = data[i][pert][order][3]
                        for entity_pair in fictitious_entities:
                            new_statement = re.sub(r"\bA\b", entity_pair[0], statement)
                            new_statement = re.sub(r"\bB\b", entity_pair[1], new_statement)
                            if pert == 'original' and order == 'asymmetric_premise':
                                # eval
                                eval_statements.append((new_statement, rignt_answer, [rignt_answer, wrong_answer]))
                                
                            elif pert == 'negation' and order == 'asymmetric_conclusion':
                                # test
                                test_statements.append((new_statement, rignt_answer, [rignt_answer, wrong_answer]))
                            else:

                                
                                # statement = data[i][pert][order][0]
                                # rignt_answer = data[i][pert][order][2]
                                # wrong_answer = data[i][pert][order][3]
                                
                                # for entity_pair in fictitious_entities:
                                #     new_statement = re.sub(r"\bA\b", entity_pair[0], statement)
                                #     new_statement = re.sub(r"\bB\b", entity_pair[1], new_statement)
                                train_statements.append((new_statement, rignt_answer, [rignt_answer, wrong_answer]))
                    

            
    #             count = 0
    #             while count < number_of_entity_trials:
    #                 print(count)
    #                 for entity_pair in fictitious_entities:
    #                     for reverse in asymetric:
    #                         if len(data[i][pert][reverse]) != 0:
    #                             right_probe = data[i][pert][reverse][0]
                            
    #                             new_statement = re.sub(r"\bA\b", entity_pair[0], right_probe)
    #                             new_statement = re.sub(r"\bB\b", entity_pair[1], new_statement)
    #                             right_probes.append(new_statement)
                                
    #                     count += 1
    #                     # right_probes.append(data[i][pert][reverse][0])
    #                     # wrong_probes.append(data[i][pert][reverse][1])
    #                     # print(data[i][pert][reverse][0])
    #                     # print(data[i][pert][reverse][1])

    # statements = []
    
    # # for statement in wrong_probes:
    # #     random.seed()
    # #     fictitious_entities = proc.generate_pairs_of_random_strings(number_of_pairs=10, 
    # #                                                             min_length=3,
    # #                                                             max_length=12,
    # #                                                             character_set=chars)
    # #     for entity_pair in fictitious_entities:
    # #         new_statement = re.sub(r"\bA\b", entity_pair[0], statement)
    # #         new_statement = re.sub(r"\bB\b", entity_pair[1], new_statement)
    # #         statements.append(new_statement)

    with open('../data/10k_perset_reduced_10/'+str(number_of_entity_trials)+'/train_sentences.txt','w') as train:
        for i, sent in enumerate(train_statements):
            train.write(sent[0])
            if i < len(train_statements) - 1:
                train.write('\n')
    with open('../data/10k_perset_reduced_10/'+str(number_of_entity_trials)+'/train_sentences_m.txt','w') as train_m:
        for i, sent in enumerate(train_statements):
            train_m.write(sent[1])
            if i < len(train_statements) - 1:
                train_m.write('\n')

    # with open('../data/10k_perset_random/eval_sentences.txt','w') as train:
    #     for i, sent in enumerate(eval_statements):
    #         train.write(sent[0])
    #         if i < len(eval_statements) - 1:
    #             train.write('\n')
    # with open('../data/10k_perset_random/eval_sentences_m.txt','w') as train_m:
    #     for i, sent in enumerate(eval_statements):
    #         train_m.write(sent[1])
    #         if i < len(eval_statements) - 1:
    #             train_m.write('\n')
    # with open('../data/10k_perset_random/test_sentences.txt','w') as train:
    #     for i, sent in enumerate(test_statements):
    #         train.write(sent[0])
    #         if i < len(test_statements) - 1:
    #             train.write('\n')
    # with open('../data/10k_perset_random/test_sentences_m.txt','w') as train_m:
    #     for i, sent in enumerate(test_statements):
    #         train_m.write(sent[1])
    #         if i < len(test_statements) - 1:
    #             train_m.write('\n')
    # with open('../data/10k_perset_random/config.txt','w') as config:
    #     for i, sent in enumerate(test_statements):
    #         config.write(str(sent[2]))
    #         if i < len(test_statements) - 1:
                # config.write('\n')
    
    # with open('../data/10k_fixed/'+number_of_entity_trials+'/test_sentences.txt','w') as test:


if __name__ == "__main__":
    main()



