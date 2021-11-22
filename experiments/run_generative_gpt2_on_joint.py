import json
import sys
import logging
import random
import string
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
import experiment_utils as utils

import pathlib
# sys.path.append('../')
curr_path = str(pathlib.Path().absolute())
home = curr_path.replace('/experiments','')
sys.path.append(home+'/data_creation')
import pre_processing_utils as proc

import csv
from collections import defaultdict


def run_pipeline(model, tokenizer, possible_chars, sentences, number_of_trials, logger):
    
    dataset = proc.prepare_truism_data_for_sentence_scoring(sentences,
                                                       possible_chars,
                                                       tokenizer,
                                                       number_of_trials)

    logger.info("finished creating dataset")

    perf = utils.generative_truism_reasoning_test(dataset, model, torch.cuda.is_available(), logger)

    logger.info("finished evaluating dataset")
    
    
    output_df = utils.convert_bi_statistic_results_into_df(perf)

    return output_df

def main():
    random.seed()
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger = logging.getLogger(__name__)
    chars = list(string.ascii_uppercase.replace("A", "").replace("I", "").replace("U", ""))
    # number_of_trials = 10

    tokenizer = GPT2Tokenizer.from_pretrained('../../model/gpt-2/'+str(checkpoint_dir))
    model = GPT2LMHeadModel.from_pretrained("../../model/gpt-2/"+str(checkpoint_dir))

    test_sents = {}

    with open("../data/generation_test_data/gpt2/physical_data_sentences.json", "r") as f:
        physical_sents = json.load(f)
    for key, val in physical_sents.items():
        test_sents[key+'-P'] = val
    logger.info("finished reading in physical data")

        
    with open("../data/generation_test_data/gpt2/material_data_sentences.json", "r") as f:
        material_sents = json.load(f)
    for key, val in material_sents.items():
        test_sents[key+'-M'] = val 

    logger.info("finished reading in material data")

    with open("../data/generation_test_data/gpt2/social_data_sentences.json", "r") as f:
        social_sents = json.load(f)
    for key, val in social_sents.items():
        test_sents[key+'-S'] = val 
    logger.info("finished reading in social data")

    with open("../data/generation_test_data/gpt2/temporal_data_sentences.json", "r") as f:
        temporal_sents = json.load(f)
    for key, val in temporal_sents.items():
        test_sents[key+'-original-T'] = val 
    logger.info("finished reading in temporal data")

    with open("../data/10k_GPT2/test_pairs.json", "r") as f:
        new_sentences = json.load(f)

    test_index = []
    
    for i in range(170):
        test_index.append(str(i))
        
    
    for key, val in new_sentences.items():
        if key.split('-')[0] in test_index:
            test_sents[key+'-10k'] = val

    logger.info("finished reading in 10k data")

    output_df = run_pipeline(model=model,
                             tokenizer=tokenizer,
                             possible_chars=chars, 
                             sentences=test_sents, 
                             number_of_trials=int(number_of_trials),
                             logger=logger)

    output_df.to_csv("../data/zeroshot_joint/gpt2_{}.csv".format(torch.cuda.initial_seed()),index=False)

    logger.info("finished saving dataset results")



def get_average():
    columns = defaultdict(list) 

    with open("../data/zeroshot_joint/gpt2_{}.csv".format(torch.cuda.initial_seed()), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader: 
            for (k,v) in row.items(): 
                columns[k].append(v) 
                                    


    total_score = 0
    for score in columns['avg_binary_score']:
        total_score += float(score)

    average_score = total_score / len(columns['avg_binary_score'])
    print(average_score)

if __name__ == "__main__":
    
    checkpoint_dir = sys.argv[1]
    number_of_trials = sys.argv[2]
    main()
    get_average()
