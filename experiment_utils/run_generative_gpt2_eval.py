import json
import sys
import logging
import random
import string
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
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

def main():
    random.seed(1012)
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger = logging.getLogger(__name__)
    chars = list(string.ascii_uppercase.replace("A", "").replace("I", "").replace("U", ""))
    number_of_trials = 1

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    with open("../data/generation_test_data/100k_test_pairs.json", "r") as f:
        new_sentences = json.load(f)
        

    logger.info("finished reading in 100k test data")

    output_df = run_pipeline(model=model,
                             tokenizer=tokenizer,
                             possible_chars=chars, 
                             sentences=new_sentences, 
                             number_of_trials=number_of_trials,
                             logger=logger)

    output_df.to_csv("../data/100k_GPT2/without_ft_{}.csv".format(number_of_trials),
                     index=False)

    logger.info("finished saving 100k dataset results")

if __name__ == "__main__":
    main()
