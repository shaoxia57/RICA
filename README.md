1. For [Leaderboard](https://inklab.usc.edu/RICA/#exp) submission:
    1. Zero-Shot Setting:
        - Training data : None
        - Test data: `data/joint_test_set/test_sentences.txt`
    2. Fine-Tuning Setting:
        - Training data : `data/10k/`
        - Test data: `data/joint_test_set/test_sentences.txt`


2. All data is in `data`:
    - We prepared the json format to better help users to understand the structures of our probe sets, i.e., which perturbation corresponds to each statement, what is the logical form, etc. For actual testing/training files, we've replaced the A/B with random entities and are in txt format.
    1. Full 253k data (noisy) in json format
         - `data/RICA_253k_axiom2set.jsonl`
    2. Human-Verified 10k data in json format
         - `data/RICA_10k_axiom2set.jsonl`
    3. Test Data
         - Easy test set (from human-verified generated set) : `data/easy_test_sentences.txt`
         - Hard test set (from human-curated set) : `data/hard_test_sentences.txt`
         - Joint test set : `data/joint_test_set/test_sentences.txt`
               

3. Testing Models
    - Besides the following scripts, we also prepared a Jupyter notebook in `experiments/Probing_Examples.ipynb` that use the most up-to-date Huggingface pipeline for masked word prediction.
    1. For zeroshot
        - BERT `experiments/eval_bert_zeroshot.py`
    
            `python eval_bert_zeroshot.py test_dir_name test_name(easy/hard/joint) filename seed`
       
            `e.g. python eval_bert_zeroshot.py joint_test_set joint bert-large-42 42`
        - RoBERTa `experiments/eval_roberta_zeroshot.py`
    
            `python eval_roberta_zeroshot.py test_dir_name test_name(easy/hard/joint) filename seed`
       
            `e.g. python eval_roberta_zeroshot.py joint_test_set joint roberta-large-42 42`
        - GPT2 `experiments/eval_gpt2_zeroshot.py` 
    
            `python eval_gpt2_zeroshot.py`
        
    2. For finetuned models
        - BERT `experiments/eval_bert_finetuned.py`
    
            `python eval_bert_finetuned.py test_data_dir model_dir output_dir #of_novel_entities model_name`
       
            `e.g. python eval_bert_finetuned.py human_curated_set 10k 10k_fine_tuned 1 bert-large-42`
        - RoBERTa `experiments/eval_roberta_finetuned.py`
    
            `python eval_roberta_finetuned.py test_data_dir model_dir output_dir #of_novel_entities model_name`
       
            `e.g. python eval_roberta_finetuned.py human_curated_set 10k 10k_fine_tuned 1 robert-large-42`
        - GPT2 `experiments/run_generative_gpt2_on_easy.py` `experiments/run_generative_gpt2_on_hard.py` `experiments/run_generative_gpt2_on_joint.py`
    
            `python run_generative_gpt2_on_easy.py #_of_novel_entities`
       
            `e.g. python run_generative_gpt2_on_easy.py 5`


   
4. Finetuning BERT/RoBERTa for MWP
    1. Finetuning code `train_mlm.py` is in `happy-transformer/examples/`
        
        `python train_mlm.py training_data_directory #_of_novel_entities output_filename model_name seed_number`
       
       `e.g. python train_mlm.py 10k 10 roberta-large-42 roberta-large 42`
        
    2. After finetuning, you can get the average binary score using `experiments/test_mlm.py`
    
        `python test_mlm.py training_data_directory filename.txt`
       
       `e.g. python test_mlm.py 10k 10 10_roberta-large-42.txt`
 
5. Finetuning GPT2
    1. You can finetune GPT2 using `experiments/fine_tune_GPT-2.sh`
    
