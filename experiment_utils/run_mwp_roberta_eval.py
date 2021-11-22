import json
import logging
import random
import string
import sys
import torch
import experiment_utils as utils

sys.path.append('../')
from dataset_creation import pre_processing_utils as proc

def run_pipeline(model, fictitious_entities, sentences, config, number_of_entity_trials, logger, temporal=False):
    if not temporal:
        dataset = proc.prepare_masked_instances(sentences=sentences, 
                                                config=config, 
                                                fictitious_entities=fictitious_entities,
                                                num_entity_trials=number_of_entity_trials)
    else:
        dataset = proc.prepare_masked_instances_temporal(sentences=sentences, 
                                                         config=config, 
                                                         fictitious_entities=fictitious_entities,
                                                         num_entity_trials=number_of_entity_trials)

    logger.info("finished creating dataset")

    perf = utils.fair_seq_masked_word_prediction(masked_examples=dataset,
                                                 model=model,
                                                 gpu_available=torch.cuda.is_available(),
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

    roberta = torch.hub.load(github='pytorch/fairseq', model='roberta.large')
    roberta.eval()

    fictitious_entities = proc.generate_pairs_of_random_strings(number_of_pairs=100, 
                                                                min_length=3,
                                                                max_length=12,
                                                                character_set=chars)

    with open("../data/truism_data/physical_data_sentences_2.json", "r") as f:
        physical_sents = json.load(f)
        
    with open("../data/truism_data/physical_data_2.json", "r") as f:
        physical_config = json.load(f)

    logger.info("finished reading in physical data")

    output_df = run_pipeline(model=roberta, 
                             fictitious_entities=fictitious_entities, 
                             sentences=physical_sents, 
                             config=physical_config, 
                             number_of_entity_trials=number_of_entity_trials,
                             logger=logger)

    output_df.to_csv("../data/masked_word_result_data/roberta-large/physical_perf_2_{}.csv".format(number_of_entity_trials),
                     index=False)

    logger.info("finished saving physical dataset results")

        
    with open("../data/truism_data/material_data_sentences_2.json", "r") as f:
        material_sents = json.load(f)
        
    with open("../data/truism_data/material_data_2.json", "r") as f:
        material_config = json.load(f)

    logger.info("finished reading in material data")

    output_df = run_pipeline(model=roberta, 
                             fictitious_entities=fictitious_entities, 
                             sentences=material_sents, 
                             config=material_config, 
                             number_of_entity_trials=number_of_entity_trials,
                             logger=logger)

    output_df.to_csv("../data/masked_word_result_data/roberta-large/material_perf_2_{}.csv".format(number_of_entity_trials),
                     index=False)

    logger.info("finished saving physical material results")
        
    with open("../data/truism_data/social_data_sentences_2.json", "r") as f:
        social_sents = json.load(f)
        
    with open("../data/truism_data/social_data_2.json", "r") as f:
        social_config = json.load(f)

    logger.info("finished reading in social data")

    output_df = run_pipeline(model=roberta, 
                             fictitious_entities=fictitious_entities, 
                             sentences=social_sents, 
                             config=social_config, 
                             number_of_entity_trials=number_of_entity_trials,
                             logger=logger)

    output_df.to_csv("../data/masked_word_result_data/roberta-large/social_perf_2_{}.csv".format(number_of_entity_trials),
                     index=False)

    logger.info("finished saving social results")

    with open("../data/truism_data/temporal_data_sentences_2.json", "r") as f:
        temporal_sents = json.load(f)
        
    with open("../data/truism_data/temporal_data_2.json", "r") as f:
        temporal_config = json.load(f)

    logger.info("finished reading in temporal data")

    output_df = run_pipeline(model=roberta, 
                             fictitious_entities=fictitious_entities, 
                             sentences=temporal_sents, 
                             config=temporal_config, 
                             number_of_entity_trials=number_of_entity_trials,
                             logger=logger,
                             temporal=True)

    output_df.to_csv("../data/masked_word_result_data/roberta-large/temporal_perf_2_{}.csv".format(number_of_entity_trials),
                     index=False)

    logger.info("finished saving temporal results")

if __name__ == "__main__":
    main()
