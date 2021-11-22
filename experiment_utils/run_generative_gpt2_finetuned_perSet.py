import json
import sys
import logging
import random
import string
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
import experiment_utils as utils
import pathlib
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
    number_of_trials = 10

    tokenizer = GPT2Tokenizer.from_pretrained('../../model/gpt-2/sample_per_set_1entity/')
    model = GPT2LMHeadModel.from_pretrained("../../model/gpt-2/sample_per_set_1entity/")

    with open("../data/10k_GPT2/test_pairs.json", "r") as f:
        new_sentences = json.load(f)

    with open("../data/10k_GPT2/test_keys.json", "r") as f:
        test_keys = json.load(f)

    filtered = {}
    for key in test_keys['10k']:
        filtered[key] = new_sentences[key]

    logger.info("finished reading in 10k data")

    output_df = run_pipeline(model=model,
                             tokenizer=tokenizer,
                             possible_chars=chars, 
                             sentences=filtered, 
                             number_of_trials=number_of_trials,
                             logger=logger)

    output_df.to_csv("../data/10k_GPT2/finetuned_10k_perf_perset_1entity_{}.csv".format(number_of_trials),
                     index=False)

    logger.info("finished saving 10k dataset results")

        

if __name__ == "__main__":
    main()
