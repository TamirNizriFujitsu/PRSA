
import os
import json
import argparse
import pandas as pd
import dataset
import utils
import llm

# Phase 2 - responsible for craeting the stolen prompt + prunning considering the category attention created on phase 1 + make the stolen prompt more clear + create <best input, target output pairs>
# Input: target preview data (demo_data) + per-category attention file from model/gradient_<Category>.json.
# Output: file cthat contains the pruned stolen prompt + edited stolen prompt per category + best input+output .


def save_to_csv(df, directory, filename):
    """Save DataFrame to a CSV file, ensuring the directory exists."""
    os.makedirs(directory, exist_ok=True)
    csv_file = os.path.join(directory, filename)
    df.to_csv(csv_file, index=False)
    print(f"File saved: {csv_file}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--theme', default='demo_data_all_categories')
    parser.add_argument('--data_dir', default='demo_data')
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--beam_search', default=1, type=int)
    parser.add_argument('--pre_pruning', default=1, type=int)
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--related_words_interval', default=5, type=int)
    parser.add_argument('--out', default=None)
    parser.add_argument('--custom', action='store_true', help='Use custom scenario-specific file naming.')
    parser.add_argument('--custom_scenario', default='custom', type=str, help='Scenario suffix used when --custom is set.')
    parser.add_argument('--temperature', default=0.7, type=float)

    parser.add_argument('--target_llm_model', default="gpt-4o", type=str, help='LLM model/deployment used to generate target and stolen outputs.')
    parser.add_argument('--generator_llm_model', default="gpt-5", type=str, help='LLM model/deployment used for stolen-prompt generation.')
    parser.add_argument('--pruning_llm_model', default="gpt-4.1-mini", type=str, help='LLM model/deployment used for prompt pre-pruning.')
    parser.add_argument('--evaluation_llm_model', default="claude-sonnet-4-5", type=str, help='LLM model/deployment used for LLM-based evaluation.')
    parser.add_argument('--editor_llm_model', default="gpt-4.1-mini", type=str, help='LLM model/deployment used for LLM-based evaluation.')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = get_args()
    scenario_suffix = f"_{args.custom_scenario}" if args.custom else ""
    if args.out is None:
        os.makedirs("log", exist_ok=True)
        args.out = f"log/{args.theme}_{args.target_llm_model}{scenario_suffix}.txt"
    config = vars(args)

    model = llm.ChatGPTPredictor(config)
    merged_path = os.path.join(args.data_dir, f"demo_data_all_categories{scenario_suffix}.csv")
    if not os.path.exists(merged_path):
        raise ValueError(f"The path '{merged_path}' does not exist in your project")
    test_data = utils.load_category_record(merged_path, args.theme)

    if os.path.exists(args.out):
        os.remove(args.out)
    print(config)

    save_file_name = f"{config['theme']}_res_{args.target_llm_model}{scenario_suffix}"

    batch = test_data[0]

    gradient_path = f"model/gradient_{config['theme']}_{args.target_llm_model}{scenario_suffix}.json"
    # Loads category attention file
    with open(gradient_path, 'r') as file:
        gradient_dict = json.load(file)
    
    print("\n========Start Stealing Target Prompt========  ... ...")

    # Loading the test data
    input_data = batch['Preview Input']
    if pd.isna(input_data):
        input_data = ""
    category = batch['Category']
    print("\nGet the prompt category:", category)
    config["theme"] = category
    target_prompt = batch['Prompt']
    output_data = model.inference(input_data, target_prompt) 

    #Creating a stolen prompt cosidereing the category attention created in the first phase
    stolen_prompt = llm.generate_prompt(config, input_data, output_data, gradient_dict) 
    # Prunning the stolen prompt
    stolen_prompt = utils.prompt_pruning_google(config, stolen_prompt, input_data, output_data, model.inference) 
    stolen_prompt = utils.format_clean(stolen_prompt)
    edited_stolen_prompt = llm.edit_stolen_prompt(stolen_prompt, model=args.editor_llm_model)

    # Extracting the best input prompt for phase 3
    best_input_prompt, target_output = utils.extract_best_input_prompt(category, args.target_llm_model, scenario_suffix)

    # document the stolen prompt
    with open(args.out, 'a') as outf:
        outf.write(json.dumps(f"input_data: {input_data}") + '\n')
        outf.write(json.dumps(f"stolen_prompt: {stolen_prompt}") + '\n')
        outf.write(json.dumps(f"target_prompt: {target_prompt}") + '\n')

    df = pd.DataFrame([{
        "category": category,
        "stolen prompt": stolen_prompt,
        "edited stolen prompt": edited_stolen_prompt,
        "best_input": best_input_prompt,
        "target_output": target_output,
    }])
    save_to_csv(df, "result", f"{save_file_name}.csv")

    

    
