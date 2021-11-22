import json

with open('../data/100k_GPT2/test_pairs.json', 'r') as f:
    data = json.load(f)
    key_sets = []
    key_dict = {}
    for key in data:
        key_sets.append(key)
    
    key_dict["100k"] = key_sets

with open('../data/100k_GPT2/test_keys.json', 'w') as file:
    json.dump(key_dict, file, indent=4)