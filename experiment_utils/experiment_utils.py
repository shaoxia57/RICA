import torch
import math
import pandas as pd
from fairseq.data.data_utils import collate_tokens
# from transformers import pipeline

def generative_truism_reasoning_test(tensors, model, gpu_available, logger):
    if gpu_available:
        for key in tensors:
            for version in tensors[key]:
                for i, tensor in enumerate(tensors[key][version]):
                    tensors[key][version][i] = tensor.cuda()
        logger.info("successfully moved tensors to gpu")

        model.cuda()
        logger.info("successfully moved model to gpu")

    model.eval()

    avg_responses = {}
    with torch.no_grad():
        for j, key in enumerate(tensors):
            binary_avg_score = 0.0
            ratio_avg_score = 0.0
            num_trials = len(tensors[key]["correct"])
            for i in range(num_trials):
                right_tensor = tensors[key]["correct"][i]
                wrong_tensor = tensors[key]["incorrect"][i]
                right_answer_outputs = model(right_tensor, labels=right_tensor)
                wrong_answer_outputs = model(wrong_tensor, labels=wrong_tensor)

                right_answer_perp = math.exp(right_answer_outputs[0].item())
                wrong_answer_perp = math.exp(wrong_answer_outputs[0].item())
            
                if right_answer_perp < wrong_answer_perp:
                    binary_avg_score += 1

                ratio_avg_score += (wrong_answer_perp - right_answer_perp) / (wrong_answer_perp + right_answer_perp)

            binary_avg_score = binary_avg_score / float(num_trials)
            ratio_avg_score = ratio_avg_score / float(num_trials)

            avg_responses[key] = {"binary_score" : binary_avg_score, "ratio_score" : ratio_avg_score}
        
        if (j+1) % 240 == 0:
            logger.info("finished 10 more")

    return avg_responses

def fair_seq_masked_word_prediction(masked_examples, model, gpu_available, top_n, logger):
    if gpu_available:
        model.cuda()
        logger.info("successfully moved model to gpu")

    model.eval()

    avg_responses = {}
    for j, key in enumerate(masked_examples):
        binary_avg_score = 0
        ratio_avg_score = 0

        for example in masked_examples[key]:
            statement, right_answer, wrong_answer = example
            responses = model.fill_mask(statement, topk=top_n)
            right_pos = top_n + 1
            wrong_pos = top_n + 1
            right_score = 0
            wrong_score = 0
            done = -1
            for i in range(len(responses)):
                possible_answer = responses[i][2].strip().lower()
                if possible_answer == right_answer:
                    right_score = responses[i][1]
                    right_pos = i
                    done += 1
                if possible_answer == wrong_answer:
                    wrong_score = responses[i][1]
                    wrong_pos = i
                    done += 1
                if done > 0:
                    break
            
            binary_score = 1 if right_pos < wrong_pos else 0
            
            if right_score + wrong_score > 0:
                ratio_score = (right_score - wrong_score) / (right_score + wrong_score)
            else:
                ratio_score = -1
            
            binary_avg_score += binary_score
            ratio_avg_score += ratio_score

        binary_avg_score /= float(len(masked_examples[key]))
        ratio_avg_score /= float(len(masked_examples[key]))

        avg_responses[key] = {"binary_score" : binary_avg_score, "ratio_score" : ratio_avg_score}
        
        if (j+1) % 240 == 0:
            logger.info("finished 10 more")

    return avg_responses

def happy_transformer_masked_word_prediction(masked_examples, model, top_n, logger):

    avg_responses = {}
    for j, key in enumerate(masked_examples):
        binary_avg_score = 0
        ratio_avg_score = 0
        for example in masked_examples[key]:
            statement, right_answer, wrong_answer = example
            statement = statement.replace('<mask>','[MASK]')
            responses = model.predict_mask(statement, num_results=top_n)
            right_pos = top_n + 1
            wrong_pos = top_n + 1
            right_score = 0
            wrong_score = 0
            done = -1
            for i in range(len(responses)):
                possible_answer = responses[i]["word"].strip().lower()
                if possible_answer == right_answer:
                    right_score = responses[i]["softmax"]
                    right_pos = i
                    done += 1
                if possible_answer == wrong_answer:
                    wrong_score = responses[i]["softmax"]
                    wrong_pos = i
                    done += 1
                if done > 0:
                    break

            binary_score = 1 if right_pos < wrong_pos else 0
            
            if right_score + wrong_score > 0:
                ratio_score = (right_score - wrong_score) / (right_score + wrong_score)
            else:
                ratio_score = -1
            
            binary_avg_score += binary_score
            ratio_avg_score += ratio_score

        binary_avg_score /= float(len(masked_examples[key]))
        ratio_avg_score /= float(len(masked_examples[key]))

        avg_responses[key] = {"binary_score" : binary_avg_score, "ratio_score" : ratio_avg_score}
        
        if (j+1) % 240 == 0:
            logger.info("finished 10 more")

    return avg_responses



def albert_masked_word_prediction(masked_examples, model, tokenizer, gpu_available, top_n, logger):
#     if gpu_available:
#         model.cuda()
#         logger.info("successfully moved model to gpu")

    model.eval()

    fill_mask_pipeline = pipeline("fill-mask", model=model, tokenizer=tokenizer, topk=top_n)
    
    avg_responses = {}
    for j, key in enumerate(masked_examples):
        binary_avg_score = 0
        ratio_avg_score = 0

        for example in masked_examples[key]:
            statement, right_answer, wrong_answer = example
            statement = statement.replace('<mask>','[MASK]')
            responses = fill_mask_pipeline(statement)
            right_pos = top_n + 1
            wrong_pos = top_n + 1
            right_score = 0
            wrong_score = 0
            done = -1
            for i in range(len(responses)):
                possible_answer = tokenizer.convert_ids_to_tokens(responses[i]['token'])[1:]
                if possible_answer == right_answer:
                    right_score = responses[i]['score']
                    right_pos = i
                    done += 1
                if possible_answer == wrong_answer:
                    wrong_score = responses[i]['score']
                    wrong_pos = i
                    done += 1
                if done > 0:
                    break
            
            binary_score = 1 if right_pos < wrong_pos else 0
            
            if right_score + wrong_score > 0:
                ratio_score = (right_score - wrong_score) / (right_score + wrong_score)
            else:
                ratio_score = -1
            
            binary_avg_score += binary_score
            ratio_avg_score += ratio_score

        binary_avg_score /= float(len(masked_examples[key]))
        ratio_avg_score /= float(len(masked_examples[key]))

        avg_responses[key] = {"binary_score" : binary_avg_score, "ratio_score" : ratio_avg_score}
        
        if (j+1) % 240 == 0:
            logger.info("finished 10 more")

    return avg_responses

def fair_seq_sent_pair_classification(sentence_pairs, model, gpu_available, logger):
    if gpu_available:
        model.cuda()
        logger.info("successfully moved model to gpu")

    model.eval()

    avg_responses = {}
    counter = 0
    for key, corr_incorr_pair in sentence_pairs.items():
        avg_responses[key] = {'correct':{'label_list':[], 'avg_accuracy':-1},
                              'neutral':{'label_list':[], 'avg_accuracy':-1},
                              'incorrect':{'label_list':[], 'avg_accuracy':-1}}
        # Correct pair (true label: entailment) results
        batch = collate_tokens([model.encode(pair[0], pair[1]) for pair in corr_incorr_pair['correct']], pad_idx=1)
        logprobs = model.predict('mnli', batch)
        
        result_list = logprobs.argmax(dim=1).tolist()
        avg_accuracy = result_list.count(2)/len(result_list)

        avg_responses[key]['correct']['label_list'] = result_list
        avg_responses[key]['correct']['avg_accuracy'] = avg_accuracy

        # Neutral pair (true label: neutral) results
        batch = collate_tokens([model.encode(pair[0], pair[1]) for pair in corr_incorr_pair['neutral']], pad_idx=1)
        logprobs = model.predict('mnli', batch)
        
        result_list = logprobs.argmax(dim=1).tolist()
        avg_accuracy = result_list.count(1)/len(result_list)

        avg_responses[key]['neutral']['label_list'] = result_list
        avg_responses[key]['neutral']['avg_accuracy'] = avg_accuracy
        
        # Incorrect pair (true label: contradiction) results
        batch = collate_tokens([model.encode(pair[0], pair[1]) for pair in corr_incorr_pair['incorrect']], pad_idx=1)
        logprobs = model.predict('mnli', batch)
        
        result_list = logprobs.argmax(dim=1).tolist()
        avg_accuracy = result_list.count(0)/len(result_list)

        avg_responses[key]['incorrect']['label_list'] = result_list
        avg_responses[key]['incorrect']['avg_accuracy'] = avg_accuracy
        
        counter += 1
        if counter % 240 == 0:
            logger.info("finished 10 more")

    return avg_responses

def fair_seq_no_neutral_sent_pair_classification(sentence_pairs, model, gpu_available, logger):
    if gpu_available:
        model.cuda()
        logger.info("successfully moved model to gpu")

    model.eval()

    avg_responses = {}
    counter = 0
    for key, corr_incorr_pair in sentence_pairs.items():
        avg_responses[key] = {'correct':{'label_list':[], 'avg_accuracy':-1},
                              'incorrect':{'label_list':[], 'avg_accuracy':-1}}
        # Correct pair (true label: entailment) results
        batch = collate_tokens([model.encode(pair[0], pair[1]) for pair in corr_incorr_pair['correct']], pad_idx=1)
        logprobs = model.predict('sentence_classification_head', batch)
        
        result_list = logprobs.argmax(dim=1).tolist()
        avg_accuracy = result_list.count(1)/len(result_list)

        avg_responses[key]['correct']['label_list'] = result_list
        avg_responses[key]['correct']['avg_accuracy'] = avg_accuracy
        
        # Incorrect pair (true label: contradiction) results
        batch = collate_tokens([model.encode(pair[0], pair[1]) for pair in corr_incorr_pair['incorrect']], pad_idx=1)
        logprobs = model.predict('sentence_classification_head', batch)
        
        result_list = logprobs.argmax(dim=1).tolist()
        avg_accuracy = result_list.count(0)/len(result_list)

        avg_responses[key]['incorrect']['label_list'] = result_list
        avg_responses[key]['incorrect']['avg_accuracy'] = avg_accuracy
        
        counter += 1
        if counter % 240 == 0:
            logger.info("finished 10 more")

    return avg_responses

def convert_bi_statistic_results_into_df(result_dictionary):
    truism_numbers = []
    perturbations = []
    premises = []
    avg_binary_scores = []
    avg_ratio_scores = []
    for key in result_dictionary:
        parts = key.split("-")
        truism_numbers.append(int(parts[0]))
        perturbations.append(parts[1])
        premises.append(parts[2])
        avg_binary_scores.append(result_dictionary[key]["binary_score"])
        avg_ratio_scores.append(result_dictionary[key]["ratio_score"])

    return pd.DataFrame.from_dict({
            "truism_number"    : truism_numbers,
            "perturbation"     : perturbations,
            "premise"          : premises,
            "avg_binary_score" : avg_binary_scores,
            "avg_ratio_score"  : avg_ratio_scores
        })

def convert_bi_statistic_100k_results_into_df(result_dictionary):
    truism_numbers = []
    perturbations = []
    premises = []
    avg_binary_scores = []
    avg_ratio_scores = []
    for key in result_dictionary:
        parts = key.split("-")
        truism_numbers.append(int(parts[0]))
        avg_binary_scores.append(result_dictionary[key]["binary_score"])
        avg_ratio_scores.append(result_dictionary[key]["ratio_score"])

    return pd.DataFrame.from_dict({
            "truism_number"    : truism_numbers,
            "avg_binary_score" : avg_binary_scores,
            "avg_ratio_score"  : avg_ratio_scores
        })



def convert_fair_seq_sent_pair_results_into_df(result_dictionary):
    set_numbers = []
    perturbations = []
    asym_perturbs = []
    ent_avg_accuracy_scores = []
    neutral_avg_accuracy_scores = []
    contr_avg_accuracy_scores = []
    ent_label_list = []
    neutral_label_list = []
    contr_label_list = []
    for key in result_dictionary:
        parts = key.split("-")
        set_numbers.append(int(parts[0]))
        perturbations.append(parts[1])
        asym_perturbs.append(parts[2])
        
        ent_avg_accuracy_scores.append(result_dictionary[key]['correct']['avg_accuracy'])
        contr_avg_accuracy_scores.append(result_dictionary[key]['incorrect']['avg_accuracy'])
        neutral_avg_accuracy_scores.append(result_dictionary[key]['neutral']['avg_accuracy'])
        
        ent_label_list.append(result_dictionary[key]['correct']['label_list'])
        contr_label_list.append(result_dictionary[key]['incorrect']['label_list'])
        neutral_label_list.append(result_dictionary[key]['neutral']['label_list'])

    return pd.DataFrame.from_dict({
            "set_number"    : set_numbers,
            "perturbation"     : perturbations,
            "asym_perturbs"          : asym_perturbs,
            "ent_avg_score" : ent_avg_accuracy_scores,
            "neutral_avg_score" : neutral_avg_accuracy_scores,
            "contr_avg_score" : contr_avg_accuracy_scores,
            "ent_label_list" : ent_label_list,
            "neutral_label_list" : neutral_label_list,
            "contr_label_list" : contr_label_list
        })
