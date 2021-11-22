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
    
    with open('../data/RICA_100k_probe_sets.json','r') as file:
        data = json.load(file)
    probes_set = {}
    for i in data:
        if len(data[i]) != 0:
            right_statement = data[i][0][0]
            wrong_statement = data[i][0][1]
            right_answer = data[i][0][2]
            wrong_answer = data[i][0][3]
    for i in data:
        if len(data[i]) != 0:
            pair = {}
            right_statement = data[i][0][0]
            wrong_statement = data[i][0][1]
            pair['correct'] = right_statement
            pair['incorrect'] = wrong_statement
                        
            probes_set[i] = pair
            


    
    with open('../data/100k_GPT2/test_pairs.json', 'w') as f:
        json.dump(probes_set, f, indent=4)


if __name__ == "__main__":
    main()