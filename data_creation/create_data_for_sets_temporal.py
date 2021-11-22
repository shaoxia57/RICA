import json
import logging
import random
import string
import sys
import numpy as np
import pre_processing_utils as proc

def main():
    random.seed(1012)
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger = logging.getLogger(__name__)
    chars = string.ascii_lowercase
    number_of_entity_trials = int(num)
    fictitious_entities = proc.generate_pairs_of_random_strings(number_of_pairs=10000, 
                                                                min_length=3,
                                                                max_length=12,
                                                                character_set=chars)

    # with open("../data/truism_data/physical_data_sentences_2.json", "r") as f:
    #     physical_sents = json.load(f)
        
    # with open("../data/truism_data/physical_data_2.json", "r") as f:
    #     physical_config = json.load(f)

    # logger.info("finished reading in physical data")

    # physical_sentences = proc.prep_ft_instances_for_sampling_by_sets(physical_sents, 
    #                                                                  physical_config, 
    #                                                                  fictitious_entities,
    #                                                                  number_of_entity_trials, int(num)/10)

    # with open("../data/truism_data/material_data_sentences_2.json", "r") as f:
    #     material_sents = json.load(f)
        
    # with open("../data/truism_data/material_data_2.json", "r") as f:
    #     material_config = json.load(f)

    # logger.info("finished reading in material data")

    # material_sentences = proc.prep_ft_instances_for_sampling_by_sets(material_sents, 
    #                                                                  material_config, 
    #                                                                  fictitious_entities,
    #                                                                  number_of_entity_trials, int(num)/10)

    # with open("../data/truism_data/social_data_sentences_2.json", "r") as f:
    #     social_sents = json.load(f)
        
    # with open("../data/truism_data/social_data_2.json", "r") as f:
    #     social_config = json.load(f)

    # logger.info("finished reading in social data")

    # social_sentences = proc.prep_ft_instances_for_sampling_by_sets(social_sents, 
    #                                                                social_config, 
    #                                                                fictitious_entities,
    #                                                                number_of_entity_trials, int(num)/10)

    # sentences = np.concatenate((physical_sentences, material_sentences, social_sentences))

    with open("../data/truism_data/temporal_data_sentences_2.json", "r") as f:
        temporal_sents = json.load(f)
        
    with open("../data/truism_data/temporal_data_2.json", "r") as f:
        temporal_config = json.load(f)

    logger.info("finished reading in temporal data")

    temporal_sentences = proc.prep_ft_instances_for_sampling_axioms_for_temporal(temporal_sents,
                                                                   temporal_config, 
                                                                   fictitious_entities,
                                                                   int(number_of_entity_trials))
    print(len(temporal_sentences))
    number = int(num)
    
    split_sentences, split_indicies = proc.sample_by_sets_finetuning_instances(temporal_sentences,
                                                                               train_pct=0.8,
                                                                               eval_pct=0.1,
                                                                               num=number)
    print(split_indicies)
    train_sents, eval_sents, test_sents = split_sentences
    train_sets, eval_sets, test_sets = split_indicies
    

    with open("../data/sample_by_sets/"+num+"/train_sentences.txt", "a") as f:
        for i, sent in enumerate(train_sents):
            for k, sen in enumerate(sent):
                
                f.write(sen[0]+'\n')
            #     if k < len(sen) - 1:
            #         f.write("\n")
            # if i < len(sent) - 1:
            #         f.write("\n")

    with open("../data/sample_by_sets/"+num+"/train_m_sentences.txt", "a") as f:
        for i, sent in enumerate(train_sents):
            for k, sen in enumerate(sent):
                
                f.write(sen[1]+'\n')

    with open("../data/sample_by_sets/"+num+"/train_sets.txt", "a") as f:
        f.write(str(train_sets))

    # with open("../data/sample_by_sets/eval_sentences.txt", "a") as f:
    #     for i, sent in enumerate(eval_sents):
    #         for k, sen in enumerate(sent):
                
    #             f.write(sen[0]+'\n')

    # with open("../data/sample_by_sets/eval_m_sentences.txt", "a") as f:
    #     for i, sent in enumerate(eval_sents):
    #         for k, sen in enumerate(sent):
                
    #             f.write(sen[1]+'\n')

    # with open("../data/sample_by_sets/eval_sets.txt", "a") as f:
    #     f.write(str(eval_sets))

    # with open("../data/sample_by_sets/test_sentences.txt", "a") as f:
    #     for i, sent in enumerate(test_sents):
    #         for k, sen in enumerate(sent):
                
    #             f.write(sen[0]+'\n')

    # with open("../data/sample_by_sets/test_m_sentences.txt", "a") as f:
    #     for i, sent in enumerate(test_sents):
    #         for k, sen in enumerate(sent):
                
    #             f.write(sen[1]+'\n')

    # with open("../data/sample_by_sets/test_sets.txt", "a") as f:
    #     f.write(str(test_sets))
    # with open("../data/sample_by_sets/config.txt", "a") as f:
    #     for i, sent in enumerate(test_sents):
    #         for k, sen in enumerate(sent):
                
    #             f.write(str(sen[2])+'\n')

if __name__ == "__main__":
    num = sys.argv[1]
    main()
