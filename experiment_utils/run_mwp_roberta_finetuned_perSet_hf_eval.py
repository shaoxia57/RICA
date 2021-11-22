import json
import logging
import random
import string
import sys
import torch
import experiment_utils as utils
from transformers import RobertaForMaskedLM, RobertaTokenizer
from transformers import RobertaConfig
#from happytransformer import HappyROBERTA

sys.path.append('../happy-transformer/')
from happytransformer import HappyROBERTA
sys.path.append('../')
from dataset_creation import pre_processing_utils as proc

def run_pipeline(model, tokenizer, fictitious_entities, sentences, config, number_of_entity_trials, logger):
    dataset = proc.prepare_masked_instances(sentences=sentences, 
                                            config=config, 
                                            fictitious_entities=fictitious_entities,
                                            num_entity_trials=number_of_entity_trials)    

    logger.info("finished creating dataset")

    perf = utils.happy_transformer_masked_word_prediction(masked_examples=dataset,
                                                 model=model,
                                                 top_n=100,
                                                 logger=logger)

    logger.info("finished evaluating dataset")
    
    output_df = utils.convert_bi_statistic_results_into_df(perf)

    return output_df


def main():
    random.seed(1012)
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger = logging.getLogger(__name__)
    chars = string.ascii_lowercase
    number_of_entity_trials = 10

    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    # checkpoint_path = '/home/rahul/common_sense_embedding_analysis/data/finetune_data/save_step_92160/checkpoint.pt'
    # state_dict = torch.load(checkpoint_path)["model"]
    # roberta = RobertaForMaskedLM.from_pretrained('roberta-base', state_dict=state_dict)
    
    # # Initializing a RoBERTa configuration
    # config = RobertaConfig.from_pretrained('roberta-base')
    # # Initializing a model from the configuration
    # roberta = RobertaForMaskedLM(config)
    # checkpoint_path = '/home/rahul/common_sense_embedding_analysis/data/finetune_data/save_step_92160/checkpoint.pt'
    # state_dict = torch.load(checkpoint_path)["model"]
    # roberta.load_state_dict(state_dict)

    roberta = HappyROBERTA('roberta-large')
    
    config = RobertaConfig.from_pretrained('roberta-large')
    mlm = RobertaForMaskedLM(config)
    #checkpoint_path = '/home/rahul/common_sense_embedding_analysis/data/finetune_data/save_step_92160/checkpoint.pt'
    #checkpoint_path = '/home/rahul/common_sense_embedding_analysis/data/finetune_data/roberta-base/save_step_230400/checkpoint.pt'
    #checkpoint_path = '/home/rahul/common_sense_embedding_analysis/data/finetune_data/roberta-base/roberta_base_best_sample_from_sets/checkpoint.pt'
    checkpoint_path = '../data/finetune_data/roberta-large/save_step_57000/checkpoint.pt'
    state_dict = torch.load(checkpoint_path)["model"]
    mlm.load_state_dict(state_dict)
    mlm.eval()

    roberta.mlm = mlm


    fictitious_entities = proc.generate_pairs_of_random_strings(number_of_pairs=100, 
                                                                min_length=3,
                                                                max_length=12,
                                                                character_set=chars)
    with open("../data/truism_data/physical_data_sentences_2.json", "r") as f:
        physical_sents = json.load(f)
    

    with open("../data/truism_data/physical_data_2.json", "r") as f:
        physical_config = json.load(f)

    with open("../data/finetune_data/sample_from_sets/test_keys.json", "r") as f:
        test_keys = json.load(f)

    phy_filtered = {}
    for key in test_keys['phy']:
        index = key.split("-")[0]
        ling_pert = key.split("-")[1]
        asym_pert = key.split("-")[2]
        if index not in phy_filtered.keys():
            phy_filtered[index] = {}
            phy_filtered[index][ling_pert] = {}
            phy_filtered[index][ling_pert][asym_pert] = physical_sents[index][ling_pert][asym_pert]
        elif ling_pert not in phy_filtered[index].keys():
            phy_filtered[index][ling_pert] = {}
            phy_filtered[index][ling_pert][asym_pert] = physical_sents[index][ling_pert][asym_pert]
        else:
            phy_filtered[index][ling_pert][asym_pert] = physical_sents[index][ling_pert][asym_pert]
    # physical_sents = {k: physical_sents[k] for k in ('11', '16')}
    # physical_config  = {k: physical_config[k] for k in ('11', '16')}

    logger.info("finished reading in physical data")

    output_df = run_pipeline(model=roberta, 
                             tokenizer=tokenizer,
                             fictitious_entities=fictitious_entities, 
                             sentences=phy_filtered, 
                             config=physical_config, 
                             number_of_entity_trials=number_of_entity_trials,
                             logger=logger)

    output_df.to_csv("../data/masked_word_result_data/roberta/sample_from_set/physical_perf_ft19_new_{}.csv".format(number_of_entity_trials),
                     index=False)

    logger.info("finished saving physical dataset results")

        
    with open("../data/truism_data/material_data_sentences_2.json", "r") as f:
        material_sents = json.load(f)
        
    with open("../data/truism_data/material_data_2.json", "r") as f:
        material_config = json.load(f)

    mat_filtered = {}
    for key in test_keys['mat']:
        index = key.split("-")[0]
        ling_pert = key.split("-")[1]
        asym_pert = key.split("-")[2]
        if index not in mat_filtered.keys():
            mat_filtered[index] = {}
            mat_filtered[index][ling_pert] = {}
            mat_filtered[index][ling_pert][asym_pert] = material_sents[index][ling_pert][asym_pert]
        elif ling_pert not in mat_filtered[index].keys():
            mat_filtered[index][ling_pert] = {}
            mat_filtered[index][ling_pert][asym_pert] = material_sents[index][ling_pert][asym_pert]
        else:
            mat_filtered[index][ling_pert][asym_pert] = material_sents[index][ling_pert][asym_pert]

    logger.info("finished reading in material data")

    output_df = run_pipeline(model=roberta, 
                             tokenizer=tokenizer,
                             fictitious_entities=fictitious_entities, 
                             sentences=mat_filtered, 
                             config=material_config, 
                             number_of_entity_trials=number_of_entity_trials,
                             logger=logger)

    output_df.to_csv("../data/masked_word_result_data/roberta/sample_from_set/material_perf_ft19_new_{}.csv".format(number_of_entity_trials),
                     index=False)

    logger.info("finished saving physical material results")
        
    with open("../data/truism_data/social_data_sentences_2.json", "r") as f:
        social_sents = json.load(f)
        
    with open("../data/truism_data/social_data_2.json", "r") as f:
        social_config = json.load(f)

    soc_filtered = {}
    for key in test_keys['soc']:
        index = key.split("-")[0]
        ling_pert = key.split("-")[1]
        asym_pert = key.split("-")[2]
        if index not in soc_filtered.keys():
            soc_filtered[index] = {}
            soc_filtered[index][ling_pert] = {}
            soc_filtered[index][ling_pert][asym_pert] = social_sents[index][ling_pert][asym_pert]
        elif ling_pert not in soc_filtered[index].keys():
            soc_filtered[index][ling_pert] = {}
            soc_filtered[index][ling_pert][asym_pert] = social_sents[index][ling_pert][asym_pert]
        else:
            soc_filtered[index][ling_pert][asym_pert] = social_sents[index][ling_pert][asym_pert]

    logger.info("finished reading in social data")

    output_df = run_pipeline(model=roberta, 
                             tokenizer=tokenizer,
                             fictitious_entities=fictitious_entities, 
                             sentences=soc_filtered, 
                             config=social_config, 
                             number_of_entity_trials=number_of_entity_trials,
                             logger=logger)

    output_df.to_csv("../data/masked_word_result_data/roberta/sample_from_set/social_perf_ft19_new_{}.csv".format(number_of_entity_trials),
                     index=False)

    logger.info("finished saving physical social results")

if __name__ == "__main__":
    main()
