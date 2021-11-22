import requests
import json

conceptnet_api = 'http://api.conceptnet.io'


def extract_related_term(word_list):
    related_word_dict = {}
    for word in word_list:
        temp_related = []
        api_url = conceptnet_api+"/related/c/en/"+word
        related_word_obj = requests.get(api_url).json()
        for related_words in related_word_obj['related']:
            if 'en' in related_words['@id']:
                temp_related.append(related_words['@id'].strip("/c/en/"))
        
        related_word_dict[word] = temp_related

    print(related_word_dict)

def extract_synonyms(word_list):
    synonym_word_dict = {}
    for word in word_list:
        temp_word = []
        api_url = conceptnet_api+"/c/en/"+word+"?rel=r/Synonym"
        synonym_word_obj = requests.get(api_url).json()
        
        for words in synonym_word_obj['edges']:
            if 'en' in words['@id']:
                candidate_word = words['start']['label']
                if "a "+word != candidate_word and "A "+word != candidate_word and candidate_word.lower() != word: 
                    temp_word.append(words['start']['label'])
        
        synonym_word_dict[word] = temp_word

    print(synonym_word_dict)

def extract_sub_events(verb_list):
    sub_event_dict = {}
    pre_event_dict = {}
    sub_index = 0
    pre_index = 0
    for i in range(len(verb_list)):

        temp_for_subevent = {'cause':'', 'result':[]}
        temp_for_pre = {'cause':[], 'result':''}
        verb = verb_list[i].strip("\n")

        api_url = conceptnet_api+"/c/en/"+verb
        sub_event_obj = requests.get(api_url).json()
        
        for word_id in sub_event_obj['edges']:
            id = word_id['end']['label'].lower()
            verb = verb.replace('_', ' ')
            if word_id['rel']['label'] == 'HasSubevent':
                if verb != id and verb not in id:
                    verb = verb.replace('_', ' ')
                    temp_for_subevent['cause'] = verb
                    label = word_id['end']['label']
                    if 'you ' in label:
                        label = label.strip("you ")
                    temp_for_subevent['result'].append(label)
            elif word_id['rel']['label'] == "HasPrerequisite":
                if verb != id and verb not in id:
                    verb = verb.replace('_', ' ')
                    temp_for_pre['result'] = verb
                    label = word_id['end']['label']
                    if 'you ' in label:
                        label = label.strip("you ")
                    temp_for_pre['cause'].append(label)

                
            # elif word_id['rel']['label'] != 'HasSubevent':
            #     related_list.append(verb)

        if temp_for_subevent['cause'] != '':
            sub_index += 1
            sub_event_dict[sub_index] = temp_for_subevent
        if temp_for_pre['result'] != '':
            pre_index += 1
            pre_event_dict[pre_index] = temp_for_pre

    print(temp_for_pre)

    with open('../data/temporal_example_after.txt','w') as f:
        f.write(json.dumps(sub_event_dict))
    with open('../data/temporal_example_before.txt','w') as f2:
        f2.write(json.dumps(pre_event_dict))



            
if __name__ == "__main__":
    word_list = ["run"]
    
    with open("../data/common_verb_2.txt",'r') as file:
        verb_list = file.readlines()
    
    extract_sub_events(verb_list)
    
    # extract_sub_events(verb_list_2)
    
            



