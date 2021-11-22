export TRAIN_FILE=../data/10k_byset_reduced_50/1/train_sentences.txt
export TEST_FILE=../data/10k_byset_reduced_50/eval_sentences.txt

# CUDA_VISIBLE_DEVICES=3 python ../transformers/examples/language-modeling/run_clm.py \
#     --output_dir=../../model/gpt-2/sample_from_set/ \
#     --model_type=gpt2 \
#     --per_gpu_train_batch_size=1 \
#     --per_gpu_eval_batch_size=1 \
#     --model_name_or_path=gpt2 \
#     --do_train \
#     --train_data_file=$TRAIN_FILE \
#     --do_eval \
#     --eval_data_file=$TEST_FILE


CUDA_VISIBLE_DEVICES=1 python ../transformers/examples/language-modeling/run_clm.py \
    --model_name_or_path gpt2 \
    --train_file $TRAIN_FILE \
    --validation_file $TEST_FILE \
    --per_gpu_train_batch_size=1 \
    --per_gpu_eval_batch_size=1 \
    --do_train \
    --do_eval \
    --seed 1234 \
    --output_dir ../../model/gpt-2/10k_byset_reduced_50_1234/