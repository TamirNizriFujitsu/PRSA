import os
import json
import argparse
import pandas as pd
import utils
import llm
import scorers


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--theme', default='Email', help='The fabricated category to evaluate against.')
    parser.add_argument('--out', default=None)
    parser.add_argument('--custom', action='store_true', help='Use custom scenario-specific file naming.')
    parser.add_argument('--custom_scenario', default='custom', type=str, help='Scenario suffix used when --custom is set.')
    parser.add_argument('--temperature', default=0.7, type=float)

    parser.add_argument('--target_llm_model', default="gpt-4o", type=str, help='LLM model/deployment used to generate target and stolen outputs.')
    parser.add_argument('--evaluation_llm_model', default="gpt-4o", type=str, help='LLM model/deployment used for LLM-based evaluation.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = get_args()
    scenario_suffix = f"_{args.custom_scenario}" if args.custom else ""

    model = llm.ChatGPTPredictor({"target_llm_model": args.target_llm_model, "temperature": args.temperature})

    path = f"result/post_phase2_data_{args.target_llm_model}{scenario_suffix}.csv"
    records = utils.load_category_record(path) # all rows, no theme filter
    fabricated_category_record = utils.load_category_record(path, args.theme)[0]
    best_input = fabricated_category_record['best_input']
    target_output = fabricated_category_record['target_output']

    # Compare each category's stolen output (created by category's stolen prompt + fabricated category best input) against the fabricated (current) category's target output
    categories_diff_array = {}
    for idx, record in enumerate(records):
        category = record['category']
        edited_stolen_prompt = record['edited stolen prompt']
        stolen_output = model.inference(best_input, edited_stolen_prompt)
        diff = scorers.MetricsScorer.functional_similarity(target_output, stolen_output)
        categories_diff_array[category] = diff

    os.makedirs("model", exist_ok=True)
    out_path = f"model/diff_array_{args.theme}_{args.target_llm_model}{scenario_suffix}.json"
    with open(out_path, 'w') as f:
        json.dump(categories_diff_array, f, indent=2)
    print(f"Saved: {out_path}")
