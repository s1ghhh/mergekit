init_model="meta-llama/Llama-2-7b-chat-hf"

models=("allenai/tulu-2-dpo-7b" "garage-bAInd/Platypus2-7B" "lmsys/vicuna-7b-v1.5" "GOAT-AI/GOAT-7B-Community" "migtissera/Synthia-7B-v1.2" "NousResearch/Nous-Capybara-7B-V1" "teknium/OpenHermes-7B")

save_path='/workspace/combination_search/mergekit/combination_search/greedy_search_rank8_init'
# awk 'BEGIN { srand() }'
all_round=${#models[@]}
((all_round++))
round=0
best_model_idx_last_round=0
for ((round = 1; round < $all_round; round++)); do
    last_round=$((round - 1))
    echo "s1gh: round_"$round
    echo "s1gh: model list: "
    for item in "${models[@]}"; do
        echo "$item"
    done
    echo "s1gh: end of model list"
    best_results=0.0
    best_model_idx=''
    for ((i = 0; i < ${#models[@]}; i++)); do
        echo "s1gh: "$i ${models[i]}
        if [ "$round" -eq 1 ]; then
            echo -e "slices:\n  - sources:\n      - model: $init_model\n        layer_range: [0, 32]\n      - model: ${models[i]}\n        layer_range: [0, 32]\nmerge_method: slerp\nbase_model: $init_model\nparameters:\n  t:\n    - filter: self_attn\n      value: 0.5\n    - filter: mlp\n      value: 0.5\n    - value: 0.5\ndtype: bfloat16" > slerp${round}_${i}.yml
        else
            echo -e "slices:\n  - sources:\n      - model: $save_path/slerp${last_round}_${best_model_idx_last_round}\n        layer_range: [0, 32]\n      - model: ${models[i]}\n        layer_range: [0, 32]\nmerge_method: slerp\nbase_model: $save_path/slerp${last_round}_${best_model_idx_last_round}\nparameters:\n  t:\n    - filter: self_attn\n      value: 0.5\n    - filter: mlp\n      value: 0.5\n    - value: 0.5\ndtype: bfloat16" > slerp${round}_${i}.yml
        fi
        # merge
        CUDA_VISIBLE_DEVICES=0 mergekit-yaml slerp${round}_${i}.yml $save_path/slerp${round}_${i} --cuda --random-seed 42
        # eval
        CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 21028  -m lm_eval --model hf \
            --model_args pretrained=$save_path/slerp${round}_${i},trust_remote_code=True \
            --tasks mmlu \
            --num_fewshot 5 \
            --batch_size 1 \
            --output_path ${save_path}/5shot_mmlu_slerp${round}_${i}.json >> ${save_path}/5shot_mmlu_slerp${round}_${i}.out
        result=$(jq -r '.results.mmlu."acc,none"' ${save_path}/5shot_mmlu_slerp${round}_${i}.json)
        result_float=$(echo "$result" | bc)
        echo "s1gh: result: "$result
        if (( $(echo "$result_float >= $best_results" | bc -l) )); then
            best_results=$result_float
            best_model_idx=$i
        fi
    done
    echo "s1gh: best result " $best_results
    echo "s1gh: remove best model "${models[best_model_idx]}
    best_model_idx_last_round=$best_model_idx

    elements_after_deleted=("${models[@]:$((best_model_idx + 1))}")
    for ((i=best_model_idx; i<${#models[@]}-1; i++)); do
    models[$i]=${elements_after_deleted[$((i - best_model_idx))]}
    done
    unset models[${#models[@]}-1]
    echo "====================="
    # fi

done

