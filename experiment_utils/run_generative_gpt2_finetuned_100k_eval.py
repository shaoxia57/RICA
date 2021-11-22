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

def main(checkpoint_dir,number_of_trials):
    random.seed(1012)
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger = logging.getLogger(__name__)
    chars = list(string.ascii_uppercase.replace("A", "").replace("I", "").replace("U", ""))
    # number_of_trials = 10

    tokenizer = GPT2Tokenizer.from_pretrained('../../model/gpt-2/'+str(checkpoint_dir))
    model = GPT2LMHeadModel.from_pretrained("../../model/gpt-2/"+str(checkpoint_dir))


    with open("../data/generation_test_data/100k_test_pairs.json", "r") as f:
        test_sents = json.load(f)

    # test_index = []
    # for i in range(767,1706,1):
    #     test_index.append(str(i))
    
    # test_sents = {}
    # for key, val in new_sentences.items():
    #     if key.split('-')[0] in test_index:
    #         test_sents[key] = val

    # logger.info("finished reading in 10k data")


        
    # with open("../data/generation_test_data/material_data_sentences.json", "r") as f:
    #     material_sents = json.load(f)
    
    # test_index = []
    # for i in range(10, 20, 1):
    #     test_index.append(str(i))
    # for key, val in material_sents.items():
    #     if key.split('-')[0] in test_index:
    #         test_sents[key+'-M'] = val

    # logger.info("finished reading in material data")

        
    # with open("../data/generation_test_data/social_data_sentences.json", "r") as f:
    #     social_sents = json.load(f)
    # for key, val in social_sents.items():
        
    #     test_sents[key+'-S'] = val


    # logger.info("finished reading in social data")

    # with open("../data/generation_test_data/temporal_data_sentences.json", "r") as f:
    #     temporal_sents = json.load(f)
    # for key, val in temporal_sents.items():
    #     test_sents[key+'T'] = val


    # logger.info("finished reading in temporal data")

    # with open("../data/generation_test_data/100k_test_pairs.json","w") as f:
    #     json.dump(test_sents, f, indent=4)

    output_df = run_pipeline(model=model,
                             tokenizer=tokenizer,
                             possible_chars=chars, 
                             sentences=test_sents, 
                             number_of_trials=int(1),
                             logger=logger)
    
    output_df.to_csv("../data/100k_GPT2/{}_finetuned_{}.csv".format(number_of_trials, str(checkpoint_dir)),
                     index=False)

    logger.info("finished saving dataset results")


        

if __name__ == "__main__":
    checkpoint_dir = sys.argv[1]
    number_of_trials = sys.argv[2]
    main(checkpoint_dir,number_of_trials)
