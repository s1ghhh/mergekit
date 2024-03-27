init_model="allenai/tulu-2-dpo-7b"

models=("garage-bAInd/Platypus2-7B" "lmsys/vicuna-7b-v1.5" "GOAT-AI/GOAT-7B-Community" "migtissera/Synthia-7B-v1.2" "NousResearch/Nous-Capybara-7B-V1" "teknium/OpenHermes-7B" "meta-llama/Llama-2-7b-chat-hf")

root_path='/workspace/combination_search/mergekit/combination_search'
save_path=$root_path'/1_init_by_dissim'
# awk 'BEGIN { srand() }'
all_round=${#models[@]}
((all_round++))
round=0
best_model_idx_last_round=0
for ((round = 1; round < $all_round; round++)); do
    last_round=$((round - 1))
    echo "round_"$round
    echo "begin of model list: "
    for item in "${models[@]}"; do
        echo "$item"
    done
    echo "end of model list"

    # Calculate similarity in model zoo
    best_results=2.0
    best_model_idx=''
    for ((i = 0; i < ${#models[@]}; i++)); do
        echo "get sim of "$i ${models[i]}

        if [ "$round" -eq 1 ]; then
            python $root_path/get_sim.py --model_a $init_model --model_b ${models[i]} --save_dir $save_path/sim_${round}_${i}.json
        else
            python $root_path/get_sim.py --model_a $save_path/slerp${last_round}_best_fit --model_b ${models[i]} --save_dir $save_path/sim_${round}_${i}.json
        fi

        #
        result=$(jq -r '.sim_all' ${root_path}/sim_${round}_${i}.json)
        result_float=$(echo "$result" | bc)
        echo "s1gh: result: "$result
        if (( $(echo "$result_float <= $best_results" | bc -l) )); then
            best_results=$result_float
            best_model_idx=$i
        fi

    done
    echo "best result " $best_results
    echo "remove best model "${models[best_model_idx]}
    best_model_idx_last_round=$best_model_idx

    # make yml
    if [ "$round" -eq 1 ]; then
        echo -e "slices:\n  - sources:\n      - model: $init_model\n        layer_range: [0, 32]\n      - model: ${models[best_model_idx]}\n        layer_range: [0, 32]\nmerge_method: slerp\nbase_model: $init_model\nparameters:\n  t:\n    - filter: self_attn\n      value: 0.5\n    - filter: mlp\n      value: 0.5\n    - value: 0.5\ndtype: bfloat16" > $save_path/slerp${round}.yml
    else
        echo -e "slices:\n  - sources:\n      - model: $save_path/slerp${last_round}_best_fit\n        layer_range: [0, 32]\n      - model: ${models[best_model_idx]}\n        layer_range: [0, 32]\nmerge_method: slerp\nbase_model: $save_path/slerp${last_round}_best_fit\nparameters:\n  t:\n    - filter: self_attn\n      value: 0.5\n    - filter: mlp\n      value: 0.5\n    - value: 0.5\ndtype: bfloat16" > $save_path/slerp${round}.yml
    fi

    # merge
    CUDA_VISIBLE_DEVICES=0 mergekit-yaml $save_path/slerp${round}.yml $save_path/slerp${round} --cuda --random-seed 42

    # drop the model included in this round
    elements_after_deleted=("${models[@]:$((best_model_idx + 1))}")
    for ((i=best_model_idx; i<${#models[@]}-1; i++)); do
    models[$i]=${elements_after_deleted[$((i - best_model_idx))]}
    done
    unset models[${#models[@]}-1]
    echo "====================="


done

