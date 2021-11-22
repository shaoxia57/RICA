import os
import sys
import json
import pathlib
curr_path = str(pathlib.Path().absolute())
home = curr_path.replace('/happy-transformer/examples','')
sys.path.append(home+'/happy-transformer/happytransformer')

from happy_bert import HappyBERT
from happy_roberta import HappyROBERTA

import logging
import string
import torch
from tqdm import tqdm
sys.path.append(home+'/data_creation')
import pre_processing_utils as proc
sys.path.append(home+'/experiments')
import experiment_utils as utils
import numpy
import ast

def model_init():

    logger = logging.getLogger(__name__)

    
    if sys.argv[4] == 'bert-base':
        model_name = "bert-base-uncased"
        model = HappyBERT(model_name)
    elif sys.argv[4] == 'roberta-base':
        model_name = 'roberta-base'
        model = HappyROBERTA(model_name)
    elif sys.argv[4] == 'roberta-large':
        model_name = 'roberta-large'
        model = HappyROBERTA(model_name)
    elif sys.argv[4] == 'bert-large':
        model_name = 'bert-large-uncased'
        model = HappyBERT(model_name)

    # batch size = bert 8 roberta 4
    if model_name == "bert-base-uncased":
        bs = 8
    elif model_name == 'roberta-base':
        bs = 4
    elif model_name == 'roberta-large':
        bs = 1
    elif model_name == 'bert-large-uncased':
        bs = 2
    word_prediction_args = {
    "batch_size": bs,

    "epochs": 20,

    "lr": 1e-5,

    "adam_epsilon": 1e-6

    }

    model.init_train_mwp(word_prediction_args)
    return model, model_name, logger



def train(model, model_name, logger, seed):
    result_list = []

    train_path = '../../data/'+dir_name+'/'+num_of_entities+'/train_sentences.txt'
    train_masked_path = '../../data/'+dir_name+'/'+num_of_entities+'/train_sentences_m.txt'
    eval_path = '../../data/'+dir_name+'/eval_sentences.txt'
    eval_masked_path = '../../data/'+dir_name+'/eval_sentences_m.txt'
    output_dir = home.replace('/RICA','')+'/model/'+dir_name+'/'+num_of_entities+'/'+filename+'/'
    print(output_dir)
    model.train_mwp(train_path, eval_path, train_masked_path, eval_masked_path, output_dir, seed)

    return model
    
def evaluation(model):
    result_list = []
    config_list = []
    config_path = '../../data/'+dir_name+'/config.txt'

    with open(config_path, "r") as f:
        config = f.readlines()
    
        
    test_path = '../../data/'+dir_name+'/test_sentences.txt'
    test_m_path = '../../data/'+dir_name+'/test_sentences_m.txt'

    with open(test_path,'r') as test_file:
        sentence = test_file.readlines()
    with open(test_m_path,'r') as test_m_file:
        mask = test_m_file.readlines()
    
    for k in tqdm(range(len(sentence))):
        mask_word = mask[k].strip("\n")
        sentence_w_mask = sentence[k].replace(mask_word, "[MASK]")
        options = ast.literal_eval(config[k])
        result = model.predict_mask(sentence_w_mask, options=options, num_results=2)
        result_list.append(str(result))
        

    
    with open('../../data/'+dir_name+'/result/'+num_of_entities+'_'+filename+'.txt', 'w') as result_file:
        result_file.write('\n'.join(result_list))


if __name__ == "__main__":
    dir_name = sys.argv[1]
    num_of_entities = sys.argv[2]
    filename = sys.argv[3] # model name + seed number
    seed = sys.argv[5]
    model, model_name, logger = model_init()
    model = train(model, model_name, logger, seed)
    evaluation(model)

# python train_mlm.py 100k 1 roberta-large-42 roberta-large 42
