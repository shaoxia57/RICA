import json
import logging
import random
import string
import sys
import numpy as np
import pre_processing_utils as proc
import re

def main():
    random.seed(1012)
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger = logging.getLogger(__name__)
    chars = string.ascii_lowercase
    probes_set = {}
    with open('../data/truism_data/temporal_data_sentences_2.json','r') as file:
        data = json.load(file) 
        
        perturbations = ['original','negation','antonym','paraphrase','paraphrase_inversion','negation_antonym','negation_paraphrase','negation_paraphrase_inversion']
        for i in data:
            for pert in perturbations:
                pair = {}
                right_statement = data[i][pert]
                
                pair['correct'] = right_statement
                
                if 'after' in right_statement:
                    wrong_statement = right_statement.replace('after','before')
                elif 'before' in right_statement:
                    wrong_statement = right_statement.replace('before','after')
                pair['incorrect'] = wrong_statement
                
                probes_set[i+'-'+pert] = pair
                print(probes_set)
                print('----')

    with open('../data/generation_test_data/gpt2/temporal_data_sentences.json', 'w') as f:
        json.dump(probes_set, f, indent=4)


if __name__ == "__main__":
    main()