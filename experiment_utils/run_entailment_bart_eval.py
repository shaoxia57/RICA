import json
import logging
import random
import string
import sys
import torch
import experiment_utils as utils

sys.path.append('../')
from dataset_creation import pre_processing_utils as proc

def run_pipeline(model, fictitious_entities, sentences, number_of_entity_trials, logger):
    dataset = proc.prepare_sentence_pair(sentences=sentences, 
                                         fictitious_entities=fictitious_entities,
                                         num_entity_trials=number_of_entity_trials)    

    logger.info("finished creating dataset")

    perf = utils.fair_seq_sent_pair_classification(sentence_pairs=dataset,
                                                   model=model,
                                                   gpu_available=torch.cuda.is_available(),
                                                   logger=logger)

    logger.info("finished evaluating dataset")
    
    output_df = utils.convert_fair_seq_sent_pair_results_into_df(perf)

    return output_df

def main():
    random.seed(1012)
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger = logging.getLogger(__name__)
    chars = string.ascii_lowercase
    number_of_entity_trials = 10

    bart = torch.hub.load(github='pytorch/fairseq', model='bart.large.mnli')

    fictitious_entities = proc.generate_pairs_of_random_strings(number_of_pairs=100, 
                                                                min_length=3,
                                                                max_length=12,
                                                                character_set=chars)

    with open("../data/entailment_test_data/physical_data_sentences.json", "r") as f:
        physical_sents = json.load(f)
    

    logger.info("finished reading in physical data")

    output_df = run_pipeline(model=bart, 
                             fictitious_entities=fictitious_entities, 
                             sentences=physical_sents, 
                             number_of_entity_trials=number_of_entity_trials,
                             logger=logger)

    output_df.to_csv("../data/entailment_result_data/bart/physical_entail_perf_{}.csv".format(number_of_entity_trials),
                     index=False)

    logger.info("finished saving physical results")

        
    with open("../data/entailment_test_data/material_data_sentences.json", "r") as f:
        material_sents = json.load(f)

    logger.info("finished reading in material data")

    output_df = run_pipeline(model=bart, 
                             fictitious_entities=fictitious_entities, 
                             sentences=material_sents, 
                             number_of_entity_trials=number_of_entity_trials,
                             logger=logger)

    output_df.to_csv("../data/entailment_result_data/bart/material_entail_perf_{}.csv".format(number_of_entity_trials),
                     index=False)

    logger.info("finished saving material results")
        
    with open("../data/entailment_test_data/social_data_sentences.json", "r") as f:
        social_sents = json.load(f)

    logger.info("finished reading in social data")

    output_df = run_pipeline(model=bart, 
                             fictitious_entities=fictitious_entities, 
                             sentences=social_sents,  
                             number_of_entity_trials=number_of_entity_trials,
                             logger=logger)

    output_df.to_csv("../data/entailment_result_data/bart/social_entail_perf_{}.csv".format(number_of_entity_trials),
                     index=False)

    logger.info("finished saving social results")

    with open("../data/entailment_test_data/temporal_data_sentences.json", "r") as f:
        temporal_sents = json.load(f)

    logger.info("finished reading in temporal data")

    output_df = run_pipeline(model=bart, 
                             fictitious_entities=fictitious_entities, 
                             sentences=temporal_sents,  
                             number_of_entity_trials=number_of_entity_trials,
                             logger=logger)

    output_df.to_csv("../data/entailment_result_data/bart/temporal_entail_perf_{}.csv".format(number_of_entity_trials),
                     index=False)

    logger.info("finished saving temporal results")

if __name__ == "__main__":
    main()
