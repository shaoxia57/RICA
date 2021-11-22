import json
import logging
import random
import string
import sys
import torch
import experiment_utils as utils
from transformers import RobertaForMaskedLM, RobertaTokenizer
from transformers import RobertaConfig
from happytransformer import HappyROBERTA
from tqdm import tqdm
import ast
import os
import glob
import natsort
import logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    if 'roberta-large' in file_name:
        roberta = HappyROBERTA('roberta-large')
        config = RobertaConfig.from_pretrained('roberta-large')
    elif 'roberta-base' in file_name:
        roberta = HappyROBERTA('roberta-base')
        config = RobertaConfig.from_pretrained('roberta-base')
    mlm = RobertaForMaskedLM(config)
    ckp_path = ''
    model_saved_dir = '/home/seyeon/model/'+checkpoint_dir+'/'+num_of_entities+'/'+file_name+'/' # checkpoint
    print(model_saved_dir)
    if os.path.exists(model_saved_dir):
        checkpoint_list = glob.glob(os.path.join(model_saved_dir, "save_step_*"))
        sorted_list = natsort.natsorted(checkpoint_list)
        logger.info(sorted_list)
        if os.path.exists(sorted_list[-1]+'/checkpoint.pt'):
            ckp_path = sorted_list[-1] + '/checkpoint.pt'
        else:
            ckp_path = sorted_list[-4] + '/checkpoint.pt'
    logger.info(ckp_path)
    
    state_dict = torch.load(ckp_path)["model"]
    mlm.load_state_dict(state_dict)
    mlm.eval()

    roberta.mlm = mlm




    
    result_list = []
    config_list = []
    config_path = '../data/'+test_dir_name+'/config.txt'

    with open(config_path, "r") as f:
        config = f.readlines()
    
        
    test_path = '../data/'+test_dir_name+'/test_sentences.txt'
    test_m_path = '../data/'+test_dir_name+'/test_sentences_m.txt'

    # axiom specific
    # num_test_each_axioms = int(int(num_of_entities)*0.1)
    
    with open(test_path,'r') as test_file:
        sentence = test_file.readlines()
    with open(test_m_path,'r') as test_m_file:
        mask = test_m_file.readlines()
    
    for k in tqdm(range(len(sentence))):
        mask_word = mask[k].strip("\n")

        # before_comma = sentence[k].split(',')[0]
        # after_comma = sentence[k].split(',')[1]

        # after_comma = after_comma.replace(mask_word, "[MASK]")
        # sentence_w_mask = before_comma+after_comma
        sentence_w_mask = sentence[k].replace(mask_word, "[MASK]")
        # print(sentence_w_mask)
        #axiom-specific
        # config_index = k//num_test_each_axioms
        # options = ast.literal_eval(config[config_index])

        options = ast.literal_eval(config[k])


        
        result = roberta.predict_mask(sentence_w_mask, options=options, num_results=2)
        
        result_list.append(str(result))
        

    
    with open('../data/'+output_dir_name+'/'+num_of_entities+'_'+file_name+'.txt', 'w') as result_file:
        result_file.write('\n'.join(result_list))

def evaluation():
    correct_count = 0
    mask_word = []
    result_output = []
    
    result_path = '../data/'+output_dir_name+'/'+num_of_entities+'_'+file_name+'.txt'
    with open(result_path, 'r') as result_file:
        result_list = result_file.readlines()
    print(result_path)
    for index in range(20):
        right_answer_path = '../data/'+test_dir_name+'/test_sentences_m.txt'
        with open(right_answer_path, 'r') as answer_file:
            right_answer = answer_file.readlines()
            for answer in right_answer:

                mask_word.append(answer.strip("\n"))

    for i in range(len(result_list)):
        first_answer = result_list[i].split('},')[0]
        
        if mask_word[i] in first_answer:
            answer_output = "right answer :"+ str(mask_word[i])+ "    result: "+ str(result_list[i])+"    score:"+"correct"
            result_output.append(answer_output)
            correct_count += 1
        else:
            answer_output = "right answer :"+ str(mask_word[i])+ "    result: "+ str(result_list[i])+"    score:"+"wrong"
            result_output.append(answer_output)
    accuracy = correct_count/len(result_list) * 100
    print(round(accuracy,2))
    output_dir = 'result/'+output_dir_name
    
    output_file_path = 'result/'+output_dir_name+'/'+num_of_entities+'_'+file_name+'.txt'
    
    with open(output_file_path,'w') as f:
        f.write('\n'.join(result_output))
        
    
    # accuracy = correct_count/len(result_list) * 100
    # print(round(accuracy,2))

if __name__ == "__main__":
    test_dir_name = sys.argv[1]
    checkpoint_dir = sys.argv[2]
    output_dir_name = sys.argv[3]
    num_of_entities = sys.argv[4]
    file_name = sys.argv[5]

    main()
    evaluation()

# python eval_roberta_finetuned.py joint_test_set 10k_byset joint_test_result 1 roberta-large-42
