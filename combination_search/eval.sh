model_path=''

for ((i=1; i<=7; i++))
do
#  CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 21123  -m lm_eval --model hf \
#      --model_args pretrained=$model_path/slerp${i},trust_remote_code=True \
#      --tasks winogrande \
#      --num_fewshot 5 \
#      --batch_size 1 \
#      --output_path $model_path/1.0_5shot_winogrande_model_idx_$i.json >> $model_path/1.0_model_idx_$i.out
#
#  CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 21123  -m lm_eval --model hf \
#      --model_args pretrained=$model_path/slerp${i},trust_remote_code=True \
#      --tasks hellaswag \
#      --num_fewshot 10 \
#      --batch_size 1 \
#      --output_path $model_path/1.0_10shot_hellaswag_model_idx_$i.json >> $model_path/1.0_model_idx_$i.out
#
#  CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 21123  -m lm_eval --model hf \
#      --model_args pretrained=$model_path/slerp${i},trust_remote_code=True \
#      --tasks arc_challenge \
#      --num_fewshot 25 \
#      --batch_size 1 \
#      --output_path $model_path/1.0_25shot_arc_challenge_model_idx_$i.json >> $model_path/1.0_model_idx_$i.out
#
#  CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 21123  -m lm_eval --model hf \
#      --model_args pretrained=$model_path/slerp${i},trust_remote_code=True \
#      --tasks truthfulqa \
#      --num_fewshot 0 \
#      --batch_size 1 \
#      --output_path $model_path/1.0_0shot_truthfulqa_model_idx_$i.json >> $model_path/1.0_model_idx_$i.out
#
#  CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 21123  -m lm_eval --model hf \
#      --model_args pretrained=$model_path/slerp${i},trust_remote_code=True \
#      --tasks gsm8k \
#      --num_fewshot 5 \
#      --batch_size 1 \
#      --output_path $model_path/1.0_5shot_gsm8k_model_idx_$i.json >> $model_path/1.0_model_idx_$i.out

  CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 21123  -m lm_eval --model hf \
      --model_args pretrained=$model_path/slerp${i},trust_remote_code=True \
      --tasks mmlu \
      --num_fewshot 5 \
      --batch_size 1 \
      --output_path $model_path/1.0_5shot_mmlu_model_idx_$i.json >> $model_path/1.0_model_idx_$i.out
  CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 21123  -m lm_eval --model hf \
      --model_args pretrained=$model_path/slerp${i},trust_remote_code=True \
      --tasks mmlu \
      --limit 0.1 \
      --num_fewshot 5 \
      --batch_size 1 \
      --output_path $model_path/0.1_5shot_mmlu_model_idx_$i.json >> $model_path/0.1_model_idx_$i.out
done

