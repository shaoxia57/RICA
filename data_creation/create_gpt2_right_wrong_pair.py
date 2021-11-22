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
    
    with open("../data/truism_data/RICA_10k_probe_sets.json", "r") as f:
        data = json.load(f)
        probes_set = {}
        perturbations = ['original', 'negation']
        asymetric = ['original', 'asymmetric_premise', 'asymmetric_conclusion']
    #     right_probes = []
    #     wrong_probes = []
        
        for i in data:
            for pert in perturbations:
                for order in asymetric:
                    if len(data[i][pert][order]) != 0:
                        pair = {}
                        right_statement = data[i][pert][order][0]
                        wrong_statement = data[i][pert][order][1]
                        pair['correct'] = right_statement
                        pair['incorrect'] = wrong_statement

                        probes_set[i+'-'+pert+'-'+order] = pair
    
    with open('../data/10k_GPT2/test_pairs.json', 'w') as f:
        json.dump(probes_set, f, indent=4)


if __name__ == "__main__":
    main()