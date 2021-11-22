import json
import logging
import random
import string
import sys
import numpy as np
import pre_processing_utils as proc

random.seed(42)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
chars = string.ascii_lowercase
number_of_entity_trials = 1
fictitious_entities = proc.generate_pairs_of_random_strings(number_of_pairs=number_of_entity_trials*80, 
                                                            min_length=3,
                                                            max_length=12,
                                                            character_set=chars)

print(fictitious_entities)

with open("../data/truism_data/physical_data_sentences_2.json", "r") as f:
    physical_sents = json.load(f)
        
with open("../data/truism_data/physical_data_2.json", "r") as f:
    physical_config = json.load(f)

logger.info("finished reading in physical data")

physical_sentences = proc.prep_ft_instances_for_axioms_w_perturbation(physical_sents,
                                                                    physical_config, 
                                                                    fictitious_entities,
                                                                    number_of_entity_trials)

with open("../data/truism_data/material_data_sentences_2.json", "r") as f:
    material_sents = json.load(f)
    
with open("../data/truism_data/material_data_2.json", "r") as f:
    material_config = json.load(f)

logger.info("finished reading in material data")

material_sentences = proc.prep_ft_instances_for_axioms_w_perturbation(material_sents,
                                                                    material_config, 
                                                                    fictitious_entities,
                                                                    number_of_entity_trials)

with open("../data/truism_data/social_data_sentences_2.json", "r") as f:
    social_sents = json.load(f)
    
with open("../data/truism_data/social_data_2.json", "r") as f:
    social_config = json.load(f)

logger.info("finished reading in social data")

social_sentences = proc.prep_ft_instances_for_axioms_w_perturbation(social_sents,
                                                                social_config, 
                                                                fictitious_entities,
                                                                number_of_entity_trials)

with open("../data/truism_data/temporal_data_sentences_2.json", "r") as f:
    temporal_sents = json.load(f)
    
with open("../data/truism_data/temporal_data_2.json", "r") as f:
    temporal_config = json.load(f)

logger.info("finished reading in temporal data")

temporal_sentences = proc.prep_ft_instances_for_sampling_axioms_for_temporal(temporal_sents,
                                                                temporal_config, 
                                                                fictitious_entities,
                                                                number_of_entity_trials)

axiom_sents = []
axiom_sents.append(("physical", physical_sentences))
axiom_sents.append(("material", material_sentences))
axiom_sents.append(("social", social_sentences))
axiom_sents.append(("temporal", temporal_sentences))

train_set = []
eval_set = []
test_set = []
sentences = []
for category_sents in axiom_sents:
    logger.info(category_sents[0]+"processing with "+str(number_of_entity_trials)+"entities")
    for index,sentence in enumerate(category_sents[1]):
        # if category_sents[0] == 'physical' or category_sents[0] == 'material':
        
        category = category_sents[0]

        
            # train_ratio = int(len(category_sents[1][index]) * 0.8)
            # eval_ratio = int(len(category_sents[1][index]) * 0.1)
            # test_ratio = int(len(category_sents[1][index]) * 0.1)

        # train_sents = sentence
            # train_sents = sentence[2:2+train_ratio]
            # eval_sent = sentence[train_ratio:train_ratio+eval_ratio]
            # test_sent = sentence[train_ratio+eval_ratio:train_ratio+eval_ratio+test_ratio]
        sentences.append(sentence[0])
        # sents = sentence
        # for test in test_sent:
        #     test_set.append(test)

eval_set = sentences[0:720]
test_set = sentences[720:-1]
# print(eval_set)

test_path = '../data/100k/'+'test_sentences.txt'
test_m_path = '../data/100k/'+'test_sentences_m.txt'
config_path = '../data/100k/'+'config.txt'
eval_path = '../data/100k/'+'eval_sentences.txt'
eval_m_path = '../data/100k/'+'eval_sentences_m.txt'

with open(test_path, "a") as test_file:
    test_file.write('\n')
    for i, sent in enumerate(test_set):
        
        test_file.write(sent[0])
        if i < len(test_set) - 1:
            test_file.write("\n")
with open(test_m_path, "a") as test_m_file:
    test_m_file.write('\n')
    for i, sent in enumerate(test_set):
        test_m_file.write(sent[1])
        if i < len(test_set) - 1:
            test_m_file.write("\n")
with open(config_path, "a") as config_file:
    config_file.write('\n')
    for i, sent in enumerate(test_set):
        config_file.write(str(sent[2]))
        if i < len(test_set) - 1:
            config_file.write("\n")
with open(eval_path, "a") as eval_file:
    eval_file.write('\n')
    for i, sent in enumerate(eval_set):
        eval_file.write(sent[0])
        if i < len(eval_set) - 1:
            eval_file.write("\n")
with open(eval_m_path, "a") as eval_m_file:
    eval_m_file.write('\n')
    for i, sent in enumerate(eval_set):
        eval_m_file.write(sent[1])
        if i < len(eval_set) - 1:
            eval_m_file.write("\n")