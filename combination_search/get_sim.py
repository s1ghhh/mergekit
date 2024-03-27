from transformers import AutoModelForCausalLM
import torch
import argparse
import torch.nn.functional as F
import json


parser = argparse.ArgumentParser()
parser.add_argument("--model_a", type=str, default='TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')
parser.add_argument("--model_b", type=str, default='TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T')
parser.add_argument("--save_dir", type=str, default='11111')
parser.add_argument("--per_col", type=bool, default=True)
parser.add_argument("--delta_param", type=bool, default=False)
parser.add_argument("--seed", type=int, default=0, help="Random seed")
args = parser.parse_args()

# python /workspace/search/sim_search/get_sim.py --model_a allenai/tulu-2-dpo-7b --model_b garage-bAInd/Platypus2-7B --save_dir /workspace/search/sim_search/test.json
# models=("allenai/tulu-2-dpo-7b", "garage-bAInd/Platypus2-7B", "lmsys/vicuna-7b-v1.5", "GOAT-AI/GOAT-7B-Community", "migtissera/Synthia-7B-v1.2", "NousResearch/Nous-Capybara-7B-V1", "teknium/OpenHermes-7B", "meta-llama/Llama-2-7b-chat-hf")

l_attn = []
l_mlp = []
l_all = []

def cosine_similarity(vector1, vector2):
    dot_product = torch.dot(vector1, vector2)
    norm_vector1 = torch.norm(vector1)
    norm_vector2 = torch.norm(vector2)
    
    similarity = dot_product / (norm_vector1 * norm_vector2)
    
    return similarity


base_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
base_model_state_dict = base_model.state_dict()

# for i in range(len(models)):
#     for j in range(i+1, len(models)):


ft_model1 = AutoModelForCausalLM.from_pretrained(args.model_a)
ft_model2 = AutoModelForCausalLM.from_pretrained(args.model_b)

dic = {}
dic['model_a'] = args.model_a
dic['model_b'] = args.model_b

model_state_dict = ft_model2.state_dict()
ft_model_state_dict = ft_model1.state_dict()
sim_list_attn = []
sim_list_mlp = []
for name, param in model_state_dict.items():

    if 'self_attn' in name or 'mlp' in name:
        # print(f"Parameter name: {name}, Shape: {param.shape}")
        # sim = cosine_similarity_m(param, ft_model_state_dict[name])
        if args.delta_param:
            param_a = param - base_model_state_dict[name]
            param_b = ft_model_state_dict[name] - base_model_state_dict[name]
        else:
            param_a = param
            param_b = ft_model_state_dict[name]
        if args.per_col:
            similarity_scores = F.cosine_similarity(param_a, param_b, dim=0)
            sim = torch.mean(similarity_scores)
            # print(sim)
        else:
            sim = cosine_similarity(param_a.view(-1), param_b.view(-1))
        if 'self_attn' in name:
            sim_list_attn.append(sim.item())
        elif 'mlp' in name:
            sim_list_mlp.append(sim.item())
sim_list_all = sim_list_attn + sim_list_mlp
print(args.model_a, args.model_b)
# print(models[i], models[j])
print("sim of attn", sum(sim_list_attn)/len(sim_list_attn))
print("sim of mlp", sum(sim_list_mlp)/len(sim_list_mlp))
print("sim of all", sum(sim_list_all)/len(sim_list_all))

dic['sim_attn'] = sum(sim_list_attn)/len(sim_list_attn)
dic['sim_mlp'] = sum(sim_list_mlp)/len(sim_list_mlp)
dic['sim_all'] = sum(sim_list_all)/len(sim_list_all)

with open(args.save_dir, 'w') as f:
    json.dump(dic, f, indent=1)


