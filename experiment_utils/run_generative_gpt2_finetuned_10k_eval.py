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
home = curr_path.replace('/experiment_utils','')
sys.path.append(home+'/data_creation')
import pre_processing_utils as proc

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

def main(checkpoint_dir, number_of_trials):
    random.seed(1012)
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger = logging.getLogger(__name__)
    chars = list(string.ascii_uppercase.replace("A", "").replace("I", "").replace("U", ""))
    # number_of_trials = 10

    tokenizer = GPT2Tokenizer.from_pretrained('../../model/gpt-2/'+str(checkpoint_dir))
    model = GPT2LMHeadModel.from_pretrained("../../model/gpt-2/"+str(checkpoint_dir))

    with open("../data/10k_GPT2/test_pairs.json", "r") as f:
        new_sentences = json.load(f)

    test_index = []
    
    for i in range(170):
        test_index.append(str(i))
        
    
    
    test_sents = {}
    for key, val in new_sentences.items():
        if key.split('-')[0] in test_index:
            test_sents[key] = val

    logger.info("finished reading in 10k data")

    output_df = run_pipeline(model=model,
                             tokenizer=tokenizer,
                             possible_chars=chars, 
                             sentences=test_sents, 
                             number_of_trials=int(number_of_trials),
                             logger=logger)

    output_df.to_csv("../data/10k_GPT2/{}_finetuned_{}.csv".format(number_of_trials, str(checkpoint_dir)),
                     index=False)

    logger.info("finished saving dataset results")

        

if __name__ == "__main__":
    
    checkpoint_dir = sys.argv[1]
    number_of_trials = sys.argv[2]
    main(checkpoint_dir, number_of_trials)
