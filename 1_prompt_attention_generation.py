import pickle
import os
import csv
from tqdm import tqdm
import json
import argparse
import scorers
import dataset
import llm
import optimizers
import utils

# Phase 1 - responsible for creating the category attention file. 
# Input: many auxiliary prompts in same category (collect_data).
# Output: One cateory attention file

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
    parser.add_argument('--max_samples', default=15, type=int, help='Max collect_data samples per category (random). 0 = all.')

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

    config = vars(args)

    data_config = dict(config)
    if args.custom:
        data_config["theme"] = f"{args.theme}_{args.custom_scenario}"
    data = dataset.Datasets(data_config)
    scorer = scorers.MetricsScorer(args.evaluator, args.m)
    model = llm.ChatGPTPredictor(config)
    optimizer = optimizers.PAA(
        config, scorer, {})


    if os.path.exists(args.out):
        os.remove(args.out)

    print(config)

    with open(args.out, 'a') as outf:
        outf.write(json.dumps(config) + '\n')
    
    collect_data = data.load_collect_data()
    num_collect_data = len(collect_data)
    gradient_dict={}
    for epoch in range(args.epochs): # runs over each line from specific dataset to infer shared category traits
        for idx, batch in tqdm(enumerate(collect_data)):
            input_data = batch['Preview Input']
            target_prompt = batch['Prompt']
            # Creating ground truth output (from original system prompt + user input) by target-llm
            output_data = model.inference(input_data, target_prompt) 

            if epoch == 0 and idx == 0:
                # creates a temporary demo_data file to the scenario+task combo with the first line of data from collect_data
                demo_file = f"demo_data_{args.theme}_{args.target_llm_model}{scenario_suffix}.csv"
                with open(demo_file, "w", newline="", encoding="utf-8") as demo_f:
                    writer = csv.DictWriter(demo_f, fieldnames=["Category", "Prompt", "Input", "Output"])
                    writer.writeheader()
                    writer.writerow({
                        "Category": args.theme,
                        "Prompt": target_prompt,
                        "Input": input_data,
                        "Output": output_data,
                    })

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

            # update the attention category bu just adding 1 to the weak elements
            utils.update_gradient_dict(gradient, gradient_dict)     

    # Remove elements with low values - meaning those elements were quite similar between the outputs
    final_gradient = utils.filter_and_sort_dict(gradient_dict, num_collect_data/10)

    # Writing the attention category of each category to the model/ directory
    with open(args.out, 'a') as outf:
        outf.write(json.dumps(f"{args.theme}: {final_gradient}") + '\n')
    
    with open(f'model/gradient_{args.theme}_{args.target_llm_model}{scenario_suffix}.pkl', 'wb') as f:
        pickle.dump(final_gradient, f)

    with open(f'model/gradient_{args.theme}_{args.target_llm_model}{scenario_suffix}.json', 'w') as json_file:
        json.dump(final_gradient, json_file)
