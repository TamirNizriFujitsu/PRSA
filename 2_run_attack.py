
import os
import time
from tqdm import tqdm
import json
import argparse
import pandas as pd
import math
import dataset
import utils
import llm
import scorers
from sentence_bert import calculate_similarity_sbert

# Phase 2 - responsible for stealing and evaluating prompts on target cases considering the category attention created on phase 1
# Input: target preview data (demo_data) + per-category attention file from model/gradient_<Category>.json.
# Output: pruned stolen prompt per target case + evaluation result CSVs in result/.


def is_valid_score(x):
    return x != 0 and not math.isinf(x) and not math.isnan(x)


def save_to_csv(df, directory, filename):
    """Save DataFrame to a CSV file, ensuring the directory exists."""
    os.makedirs(directory, exist_ok=True)
    csv_file = os.path.join(directory, filename)
    df.to_csv(csv_file, index=False)
    print(f"File saved: {csv_file}")


def load_test_data_for_theme(config, theme, target_model, custom=False, custom_scenario="custom"):
    """Load exactly one category record from shared demo_data_all_categories.csv."""
    data_dir = config["data_dir"]
    custom_path = os.path.join(data_dir, f"demo_data_all_categories_{target_model}_{custom_scenario}.csv")
    model_specific_path = os.path.join(data_dir, f"demo_data_all_categories_{target_model}.csv")
    if custom and os.path.exists(custom_path):
        shared_path = custom_path
    elif os.path.exists(model_specific_path):
        shared_path = model_specific_path
    else:
        shared_path = os.path.join(data_dir, "demo_data_all_categories.csv")
    if not os.path.exists(shared_path):
        raise FileNotFoundError(f"Missing shared dataset: {shared_path}")

    df = pd.read_csv(shared_path, encoding="utf-8")
    if "Category" not in df.columns:
        raise ValueError(f"'Category' column missing in {shared_path}")

    df = df[df["Category"] == theme]
    if df.empty:
        raise ValueError(f"No record found for category '{theme}' in {shared_path}")

    if len(df) > 1:
        print(f"[Warning] Multiple records found for '{theme}'. Using first one.")
        df = df.iloc[[0]]

    return df.reset_index().to_dict("records")

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

    parser.add_argument('--evaluator', default="syntactic_similarity", type=str)
    parser.add_argument('--m', default=3, type=int, help='m is the sampling number for each user input')
    parser.add_argument('--n', default=2, type=int, help='n is the number of the test user input')
    parser.add_argument('--scorer', default="syntactic_similarity", type=str)
    parser.add_argument('--target_llm_model', default="gpt-4o", type=str, help='LLM model/deployment used to generate target and stolen outputs.')
    parser.add_argument('--generator_llm_model', default="gpt-4o-mini", type=str, help='LLM model/deployment used for stolen-prompt generation.')
    parser.add_argument('--pruning_llm_model', default="gpt-4o-mini", type=str, help='LLM model/deployment used for prompt pre-pruning.')
    parser.add_argument('--evaluation_llm_model', default="gpt-4o", type=str, help='LLM model/deployment used for LLM-based evaluation.')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = get_args()
    scenario_suffix = f"_{args.custom_scenario}" if args.custom else ""
    if args.out is None:
        os.makedirs("log", exist_ok=True)
        args.out = f"log/{args.theme}_{args.target_llm_model}{scenario_suffix}.txt"
    config = vars(args)

    scorer = scorers.MetricsScorer(args.evaluator, args.m)
    model = llm.ChatGPTPredictor(config)
    test_data = load_test_data_for_theme(
        config, args.theme, args.target_llm_model, custom=args.custom, custom_scenario=args.custom_scenario
    )

    if os.path.exists(args.out):
        os.remove(args.out)
    print(config)

    scores_list = []
    save_file_name = f"{config['theme']}_res_{args.target_llm_model}{scenario_suffix}"

    llm_based_eva_score_list = []

    for idx, batch in tqdm(enumerate(test_data)): # Loops over each row (test pair) in demo_data_all_categories.csv
        input_data = batch['Preview Input']
        print(idx)
        
        category = batch['Category']
        print("\nGet the prompt category:", category)
        config["theme"] = category

        gradient_path = f"model/gradient_{config['theme']}_{args.target_llm_model}{scenario_suffix}.json"
        # Loads category attention file
        with open(gradient_path, 'r') as file:
            gradient_dict = json.load(file)
        
        print("\n========Start Stealing Target Prompt========  ... ...")
        # timing check
        start = time.time()

        # Loading the test data
        target_prompt = batch['Prompt']
        output_data = batch['Preview Output'] 
        input_case1 = batch['Input Test Case1']
        input_case2 = batch['Input Test Case2']
        inputs = [input_case1, input_case2]
        inputs = inputs[:args.n]
        
        #Creating a stolen prompt cosidereing the category attention created in the first phase
        stolen_prompt = llm.generate_prompt(config, input_data, output_data, gradient_dict) 
        # Prunning the stolen prompt
        stolen_prompt = utils.prompt_pruning_google(config, stolen_prompt, input_data, output_data, model.inference) 
        stolen_prompt = utils.format_clean(stolen_prompt)

        # timing check
        print(f"\nStolen prompt is (took {time.time() - start:.3f} seconds): {stolen_prompt}" )

        '''
        (Optional 1)
        stolen_prompt = utils.prompt_pruning(config, stolen_prompt, input_data, output_data, model.inference)
        stolen_prompt = utils.format_clean(stolen_prompt_1)

        (Optional 2)
        stolen_prompt = utils.prompt_pruning_phrase_level(config, stolen_prompt, input_data, output_data, model.inference)
        stolen_prompt = utils.format_clean(stolen_prompt)
        '''

        #Calculating Results
        with open(args.out, 'a') as outf:
            outf.write(json.dumps(f"input_data: {input_data}") + '\n')
            outf.write(json.dumps(f"stolen_prompt: {stolen_prompt}") + '\n')
            outf.write(json.dumps(f"target_prompt: {target_prompt}") + '\n')

        # print("\n========Evaluation of Prompt Similarity======== ... ...")
        # prompt_sim_score = calculate_similarity_sbert(target_prompt, stolen_prompt)
        # print("\nPrompt similarity score is :", prompt_sim_score)


        print("\n========Evaluation of Functional Consistency======== ... ...")
        # First evaluation - evaluating the target outputs similarity for normalization baseline, to understand how similar is the target model to itself when using the real target prompt? (we should understand that before we evaluate the similarity between target outputs and stolen outputs)
        target_output_sim_score, target_outputs_per_input = scorer.evaluate_target_prompt(model.inference, inputs, target_prompt)
        if target_output_sim_score is None or target_outputs_per_input is None:
            print(f"[Warning] Skipped iteration {idx} due to failed target output generation.")
            continue

        stolen_output_sim_score = scorer.evaluate_stolen_prompt(model.inference, inputs, target_prompt, stolen_prompt, target_outputs_per_input)
        if stolen_output_sim_score is None:
            print(f"[Warning] Skipped iteration {idx} due to failed stolen output generation.")
            continue

        semantic_target_score,  syntactic_target_score, js_target_score = target_output_sim_score
        semantic_stolen_score, syntactic_stolen_score, js_stolen_score = stolen_output_sim_score

        #We normalize our results against the similarity score between target outputs, producing a similarity score in the range (0,1)
        if all(map(is_valid_score, [semantic_target_score, js_target_score])):
            semantic_sim_socre = min(semantic_stolen_score / semantic_target_score, 1)
            # Disabled for speed test:
            # syntactic_sim_socre = min(syntactic_stolen_score / syntactic_target_score, 1)
            js_sim_socre = min(js_stolen_score / js_target_score, 1)

            print("\nSemantic similarity score is:", semantic_sim_socre)
            # print("\nSyntactic similarity score is:", syntactic_sim_socre)
            print("\nStructural similarity score is:", js_sim_socre)
            with open(args.out, 'a') as outf:
                outf.write(json.dumps(f"Round [{idx+1}/{len(test_data)}] semantic similarity score: {semantic_sim_socre:.4f}") + '\n')
                # outf.write(json.dumps(f"Round [{idx+1}/{len(test_data)}] syntactic similarity score: {syntactic_sim_socre:.4f}") + '\n')
                outf.write(json.dumps(f"Round [{idx+1}/{len(test_data)}] structural similarity score: {js_sim_socre:.4f}") + '\n')
                # outf.write(json.dumps(f"Round [{idx+1}/{len(test_data)}] prompt similarity score: {prompt_sim_score:.4f}") + '\n')
                outf.write('\n\n')

        
            scores_list.append({
                'iteration': idx,
                'target prompt': target_prompt,
                "stolen prompt": stolen_prompt,
                'semantic similarity score': semantic_sim_socre,
                # 'syntactic similarity score': syntactic_sim_socre,
                'structural similarity score': js_sim_socre,
                #'prompt similarity score': prompt_sim_score
            })
        else:
            print(f"[Warning] Skipped iteration {idx} due to invalid denominator (0/inf/NaN) in target scores.")

        
        
        # print("\n======== LLM-based Multi-dimensional Evaluation======== ... ...")
        # for input_case in inputs:
        #     target_output = model.inference(input_case, target_prompt)
        #     generated_output =  model.inference(input_case, stolen_prompt)
            
        #     if target_output is None or generated_output is None:
        #         print(f"[Warning] Skipped input '{input_case}' due to inference failure.")
        #         continue

        #     llm_eva_res = llm.llm_based_evaluation(target_output, generated_output, model=args.evaluation_llm_model)
        #     try:
        #         llm_eva_scores = json.loads(llm_eva_res)
        #         print("LLM-based evaluation score: ", llm_eva_scores)
        #     except json.JSONDecodeError:
        #         llm_eva_scores = {"Accuracy":0, "Completeness":0, "Tone":0, "Sentiment":0, "Semantics":0}
        #         print("Error: Could not parse response as JSON.")
        #         print("Raw response:", llm_eva_res)

        #     llm_based_eva_score_list.append((input_case, target_prompt, stolen_prompt, 
        #                        llm_eva_scores["Accuracy"], llm_eva_scores["Completeness"],llm_eva_scores["Tone"],llm_eva_scores["Sentiment"],llm_eva_scores["Semantics"]))
    
    directory = "result"
    # df = pd.DataFrame(llm_based_eva_score_list, columns=['user input', 'target prompt', 'stolen prompt',
    #                                     'accuracy', 'completeness', 'tone', 'sentiment', 'semantics'])
    # save_to_csv(df, directory, f"llm_based_eva_{save_file_name}.csv")

    df_scores = pd.DataFrame(scores_list)
    save_to_csv(df_scores, directory, f"{save_file_name}.csv")
