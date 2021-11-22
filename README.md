1. All data is in `data`
    1. Training Data
         - 100k data : `data/100k`
         - 10k data for high resource : `data/10k_byset`
         - 10k data for low resource : `data/10k_byset_reduced_10` `data/10k_byset_reduced_20` `data/10k_byset_reduced_30` `data/10k_byset_reduced_50`
    2. Test Data
         - GPT2 : `data/10k_GPT2/test_pairs.json` `data/100k_GPT2/100k_test_json` `data/generation_test_data/gpt2`
         - Easy test set : `data/10k_byset/test_sentences.txt`
         - Hard test set : `data/80axioms_w_pert/test_sentences.txt`
         - Joint test set : `data/joint_test_set/test_sentences.txt`
               

2. Finetuning BERT/RoBERTa for MWP
    1. Finetuning code `train_mlm.py` is in `happy-transformer/examples/`
        
        `python train_mlm.py training_data_directory #_of_novel_entities output_filename model_name seed_number`
       
       `e.g. python train_mlm.py 10k_byset 10 roberta-large-42 roberta-large 42`
        
    2. After finetuning, you can get the average binary score using `experiments/test_mlm.py`
    
        `python test_mlm.py training_data_directory filename.txt`
       
       `e.g. python test_mlm.py 10k_byset 10 10_roberta-large-42.txt`
   
3. Finetuning GPT2
    1. You can finetune GPT2 using `experiments/fine_tune_GPT-2.sh`
 
4. Testing Models
    1. For finetuned models
        - BERT `experiments/eval_bert_finetuned.py`
    
            `python eval_bert_finetuned.py test_data_dir model_dir output_dir #of_novel_entities model_name`
       
            `e.g. python eval_bert_finetuned.py 80axioms_w_pert 10k_byset 10k_80_test_byset 5 bert-large-42`
        - RoBERTa `experiments/eval_roberta_finetuned.py`
    
            `python eval_roberta_finetuned.py test_data_dir model_dir output_dir #of_novel_entities model_name`
       
            `e.g. python eval_roberta_finetuned.py 80axioms_w_pert 10k_byset 10k_80_test_byset 5 robert-large-42`
        - GPT2 `experiments/run_generative_gpt2_on_easy.py` `experiments/run_generative_gpt2_on_hard.py` `experiments/run_generative_gpt2_on_joint.py`
    
            `python run_generative_gpt2_on_easy.py #_of_novel_entities`
       
            `e.g. python run_generative_gpt2_on_easy.py 5`
        
       
    2. For zeroshot
        - BERT `experiments/eval_bert_zeroshot.py`
    
            `python eval_bert_zeroshot.py test_dir_name test_name(easy/hard/joint) filename seed`
       
            `e.g. python eval_bert_zeroshot.py joint_test_set joint bert-large-42 42`
        - RoBERTa `experiments/eval_roberta_zeroshot.py`
    
            `python eval_roberta_zeroshot.py test_dir_name test_name(easy/hard/joint) filename seed`
       
            `e.g. python eval_roberta_zeroshot.py joint_test_set joint roberta-large-42 42`
        - GPT2 `experiments/eval_gpt2_zeroshot.py` 
    
            `python eval_gpt2_zeroshot.py`
    
