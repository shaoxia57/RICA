import json
import logging
import random
import string
import sys
import numpy as np
import pre_processing_utils as proc

# for entity-expansion, change number_of_entity_trials

def main():
    random.seed(1012)
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger = logging.getLogger(__name__)
    chars = string.ascii_lowercase
    number_of_entity_trials = 10
    fictitious_entities = proc.generate_pairs_of_random_strings(number_of_pairs=200, 
                                                                min_length=3,
                                                                max_length=12,
                                                                character_set=chars)
    mask_prediction_options = []
    with open("../data/truism_data/physical_data_sentences_2.json", "r") as f:
        data = json.load(f)
        physical_sents = {}
        for i in data:
            original = {}
            original_2 = {}
            original['original'] = data[i]['original']['original']
            original_2['original'] = original
            physical_sents[i]=original_2
        
    with open("../data/truism_data/physical_data_2.json", "r") as f:
        physical_config = json.load(f)
        for index in range(20):
            options = physical_config[str(index)]['premise_switch']['0']
            mask_prediction_options.append(str(options))

    logger.info("finished reading in physical data")

    physical_sentences = proc.prep_ft_instances_for_axioms(physical_sents,
                                                                     physical_config,
                                                                     fictitious_entities,
                                                                     number_of_entity_trials)


    with open("../data/truism_data/material_data_sentences_2.json", "r") as f:
        data = json.load(f)
        material_sents = {}
        for i in data:
            original = {}
            original_2 = {}
            original['original'] = data[i]['original']['original']
            original_2['original'] = original
            material_sents[i]=original_2
        
    with open("../data/truism_data/material_data_2.json", "r") as f:
        material_config = json.load(f)
        for index in range(20):
            options = material_config[str(index)]['premise_switch']['0']
            mask_prediction_options.append(str(options))
    logger.info("finished reading in material data")

    material_sentences = proc.prep_ft_instances_for_axioms(material_sents,
                                                                     material_config, 
                                                                     fictitious_entities,
                                                                     number_of_entity_trials)
    


    with open("../data/truism_data/social_data_sentences_2.json", "r") as f:
        data = json.load(f)
        social_sents = {}
        for i in data:
            original = {}
            original_2 = {}
            original['original'] = data[i]['original']['original']
            original_2['original'] = original
            social_sents[i]=original_2
        
    with open("../data/truism_data/social_data_2.json", "r") as f:
        social_config = json.load(f)
        for index in range(20):
            options = social_config[str(index)]['premise_switch']['0']
            mask_prediction_options.append(str(options))

    logger.info("finished reading in social data")

    social_sentences = proc.prep_ft_instances_for_axioms(social_sents,
                                                                   social_config, 
                                                                   fictitious_entities,
                                                                   number_of_entity_trials)

   

    with open("../data/truism_data/temporal_data_sentences_2.json", "r") as f:
        data = json.load(f)
        temporal_sents = {}
        for i in data:
            original = {}
            original_2 = {}
            original['original'] = data[i]['original']
            original_2['original'] = original
            temporal_sents[i]=original_2
    
    with open("../data/truism_data/temporal_data_2.json", "r") as f:
        temporal_config = json.load(f)
        for index in range(20):
            options = temporal_config[str(index)]['premise_switch']['0']
            mask_prediction_options.append(str(options))

    logger.info("finished reading in temporal data")

    temporal_sentences = proc.prep_ft_instances_for_axioms(temporal_sents,
                                                                   temporal_config, 
                                                                   fictitious_entities,
                                                                   number_of_entity_trials)

    # category_folder = ['physical','material', 'social','temporal']
    # for category in category_folder:
    #     sentences = category+"_sentences"
    #     print(sentences)

    axiom_sents = []
    axiom_sents.append(("physical", physical_sentences))
    axiom_sents.append(("material", material_sentences))
    axiom_sents.append(("social", social_sentences))
    axiom_sents.append(("temporal", temporal_sentences))
    
    train_set = []
    eval_set = []
    test_set = []
    for category_sents in axiom_sents:
        logger.info(category_sents[0]+"processing with "+str(number_of_entity_trials)+"entities")
        for index,sentence in enumerate(category_sents[1]):
            
            category = category_sents[0]


            train_ratio = int(len(category_sents[1][index]) * 0.8)
            eval_ratio = int(len(category_sents[1][index]) * 0.1)
            test_ratio = int(len(category_sents[1][index]) * 0.1)


            train_sents = sentence
            # eval_sent = sentence[train_ratio:train_ratio+eval_ratio]
            # test_sent = sentence[train_ratio+eval_ratio:train_ratio+eval_ratio+test_ratio]


            for train in train_sents:
                train_set.append(train)
            # for evals in eval_sent:
            #     eval_set.append(evals)
            # for test in test_sent:
            #     test_set.append(test)

    train_path = '../data/80axioms_w_pert/'+str(number_of_entity_trials)+'/train_sentences.txt'
    train_m_path = '../data/80axioms_w_pert/'+str(number_of_entity_trials)+'/train_sentences_m.txt'
    eval_path = '../data/80axioms_w_pert/'+str(number_of_entity_trials)+'/eval_sentences.txt'
    eval_m_path = '../data/80axioms_w_pert/'+str(number_of_entity_trials)+'/eval_sentences_m.txt'
    test_path = '../data/80axioms_w_pert/'+str(number_of_entity_trials)+'/test_sentences.txt'
    test_m_path = '../data/80axioms_w_pert/'+str(number_of_entity_trials)+'/test_sentences_m.txt'
    config_path = '../data/80axioms_w_pert/config.txt'
    with open(train_path, "w") as train_file:
        for i, sent in enumerate(train_set):
            # train_file.write('\n'.join(sent[0]))
            train_file.write(sent[0])
            if i < len(train_set) - 1:
                train_file.write("\n")
    with open(train_m_path, "w") as train_m_file:
        for i, sent in enumerate(train_set):
            train_m_file.write(sent[1])
            if i < len(train_set) - 1:
                train_m_file.write("\n")
    # with open(test_path, "w") as test_file:
    #     for i, sent in enumerate(test_set):
    #         test_file.write(sent[0])
    #         if i < len(test_set) - 1:
    #             test_file.write("\n")
    # with open(test_m_path, "w") as test_m_file:
    #     for i, sent in enumerate(test_set):
    #         test_m_file.write(sent[1])
    #         if i < len(test_set) - 1:
    #             test_m_file.write("\n")
    # with open(eval_path, "w") as eval_file:
    #     for i, sent in enumerate(eval_set):
    #         eval_file.write(sent[0])
    #         if i < len(eval_set) - 1:
    #             eval_file.write("\n")
    # with open(eval_m_path, "w") as eval_m_file:
    #     for i, sent in enumerate(eval_set):
    #         eval_m_file.write(sent[1])
    #         if i < len(eval_set) - 1:
    #             eval_m_file.write("\n")
    # with open(config_path, "w") as config_file:
    #     config_file.writelines('\n'.join(mask_prediction_options))


if __name__ == "__main__":
    main()



