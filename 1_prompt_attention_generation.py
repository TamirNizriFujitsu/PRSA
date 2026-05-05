import pickle
import os
from tqdm import tqdm
import json
import argparse
import scorers
import dataset
import llm
import optimizers
import utils

# Phase 1 - responsible for creating the category attention file + input prompt ranking file 
# Input: many auxiliary prompts in same category (collect_data).
# Output: One category attention file + input prompt ranking file

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--theme', default='Email', help='theme is categpry of the prompts.')
    parser.add_argument('--data_dir', default='collect_data')
    parser.add_argument('--custom', action='store_true', help='Load <theme>_<custom_scenario>.csv instead of <theme>.csv.')
    parser.add_argument('--custom_scenario', default='custom', type=str, help='Scenario suffix for custom datasets, e.g., <theme>_<custom_scenario>.csv')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--out', default=None)
    parser.add_argument('--attention_threshold', default=7.5, type=float, help='attention_threshold needs to update based on different gpt model inference.')

    parser.add_argument('--evaluator', default="semantic_similarity", type=str)
    parser.add_argument('--m', default=3, type=int, help='m is the sampling number for each user input.')
    parser.add_argument('--scorer', default="semantic_similarity", type=str)
    parser.add_argument('--target_llm_model', default="gpt-4o", type=str,
                        help='LLM model/deployment used to generate target and stolen outputs.')
    parser.add_argument('--generator_llm_model', default="gpt-4o-mini", type=str,
                        help='LLM model/deployment used for stolen-prompt generation.')
    parser.add_argument('--gradient_llm_model', default="gpt-4o-mini", type=str,
                        help='LLM model/deployment used to score gradients.')
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    scenario_suffix = f"_{args.custom_scenario}" if args.custom else ""
    if args.out is None:
        os.makedirs("log", exist_ok=True)
        args.out = f"log/{args.theme}_{args.target_llm_model}{scenario_suffix}.txt"
    if os.path.exists(args.out):
        os.remove(args.out)

    config = vars(args)
    data_config = dict(config)
    if args.custom:
        data_config["theme"] = f"{args.theme}_{args.target_llm_model}_{args.custom_scenario}"
    data = dataset.Datasets(data_config) #15 samples per cateory
    scorer = scorers.MetricsScorer(args.evaluator, args.m)
    model = llm.ChatGPTPredictor(config)
    optimizer = optimizers.PAA(config, scorer, {})

    with open(args.out, 'a') as outf:
        outf.write(json.dumps(config) + '\n')
    
    collect_data = data.load_collect_data()
    num_collect_data = len(collect_data)
    gradient_dict = {}
    prompt_gradient_scores = {}  # Maps input_data → sum of weak gradient elements. Necessary for phase 2
    for epoch in range(args.epochs): # Loops over each line from specific dataset to infer shared category traits
        for _, batch in tqdm(enumerate(collect_data)):
            input_data = batch['Preview Input']
            target_prompt = batch['Prompt']
            # Creating ground truth output (from original system prompt + user input) by target-llm
            output_data = model.inference(input_data, target_prompt) 

            # Creating a stolen prompt by generator-llm-model
            if gradient_dict == {}:
                base_stolen_prompt = llm.generate_prompt(config, input_data, output_data, gpt_model=args.generator_llm_model)
                
            else:
                base_stolen_prompt = llm.generate_prompt(config, input_data, output_data, gradient_dict, gpt_model=args.generator_llm_model) 
            
            if base_stolen_prompt == None:
                    continue
            # Creates pred output (from stolen system prompt + user input) by target-llm
            generated_output = model.inference(input_data, base_stolen_prompt) 

            # compare between the output created by the stolen prompt and output created by the target prompt and get weak elements - performed by gradient-llm-model
            gradient = optimizer.cal_gradients(generated_output, output_data)
            print("gradient: ", gradient)

            prompt_gradient_scores[input_data] = sum(gradient.values())

            # update the attention category bu just adding 1 to the weak elements
            utils.update_gradient_dict(gradient, gradient_dict)     

    # Remove elements with low values - meaning those elements were quite similar between the outputs
    final_gradient = utils.filter_and_sort_dict(gradient_dict, num_collect_data/10)

    # Writing the attention category of each category to the model/ directory and to the log
    with open(args.out, 'a') as outf:
        outf.write(json.dumps(f"{args.theme}: {final_gradient}") + '\n')
    
    with open(f'model/gradient_{args.theme}_{args.target_llm_model}{scenario_suffix}.pkl', 'wb') as f:
        pickle.dump(final_gradient, f)

    with open(f'model/gradient_{args.theme}_{args.target_llm_model}{scenario_suffix}.json', 'w') as json_file:
        json.dump(final_gradient, json_file)

    # dictionary of {<input prompt>: gradient value (sum of the elements that need attention)} for using in phase 2
    with open(f'model/prompt_scores_{args.theme}_{args.target_llm_model}{scenario_suffix}.json', 'w') as f:
        json.dump(prompt_gradient_scores, f, indent=2)
