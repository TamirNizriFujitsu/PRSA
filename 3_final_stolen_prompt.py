import os
import json
import argparse
import utils
import llm
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--theme", default="Email", help="The fabricated category to evaluate against.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--custom", action="store_true", help="Use custom scenario-specific file naming.")
    parser.add_argument("--custom_scenario", default="custom", type=str)
    parser.add_argument("--temperature", default=0.7, type=float)

    parser.add_argument("--target_llm_model", default="gpt-4o", type=str)
    parser.add_argument("--evaluation_llm_model", default="claude-sonnet-4-5", type=str)

    return parser.parse_args()


def build_stolen_outputs(records, model, best_input):
    """
    Returns:
        stolen_outputs: {"Ads": stolen_output, "Business": stolen_output, ...}
    """
    stolen_outputs = {}

    for idx, record in enumerate(records, start=1):
        category = record["category"]
        stolen_prompt = record["edited stolen prompt"]

        stolen_output = model.inference(best_input, stolen_prompt)

        stolen_outputs[category] = stolen_output

    return stolen_outputs



if __name__ == "__main__":
    args = get_args()

    scenario_suffix = f"_{args.custom_scenario}" if args.custom else ""

    model = llm.ChatGPTPredictor({
        "target_llm_model": args.target_llm_model,
        "temperature": args.temperature,
    })

    input_path = f"result/post_phase2_data_{args.target_llm_model}{scenario_suffix}.csv"

    records = utils.load_category_record(input_path)
    fabricated_record = utils.load_category_record(input_path, args.theme)[0]

    best_input = fabricated_record["best_input"]
    if pd.isna(best_input):
        best_input = ""
    target_output = fabricated_record["target_output"]

    stolen_outputs = build_stolen_outputs(
        records=records,
        model=model,
        best_input=best_input,
    )

    category_scores = llm.llm_based_outputs_comparison(
        target_output=target_output,
        stolen_outputs=stolen_outputs,
        model=args.evaluation_llm_model,
    )

    result = dict(sorted(category_scores.items(), key=lambda x: x[1], reverse=True))

    os.makedirs("model", exist_ok=True)

    out_path = args.out or f"model/diff_array_{args.theme}_{args.target_llm_model}{scenario_suffix}.json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved: {out_path}")