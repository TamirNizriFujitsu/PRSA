import os
import json
import argparse
import utils
import llm


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--theme", default="Email", help="The fabricated category to evaluate against.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--custom", action="store_true", help="Use custom scenario-specific file naming.")
    parser.add_argument("--custom_scenario", default="custom", type=str)
    parser.add_argument("--temperature", default=0.7, type=float)

    parser.add_argument("--target_llm_model", default="gpt-4o", type=str)
    parser.add_argument("--evaluation_llm_model", default="gpt-5", type=str)

    return parser.parse_args()


def build_indexed_stolen_outputs(records, model, best_input):
    """
    Returns:
        indexed_outputs: {1: stolen_output, 2: stolen_output, ...}
        index_to_category: {1: category, 2: category, ...}
    """
    indexed_outputs = {}
    index_to_category = {}

    for idx, record in enumerate(records, start=1):
        category = record["category"]
        stolen_prompt = record["edited stolen prompt"]

        stolen_output = model.inference(best_input, stolen_prompt)

        indexed_outputs[idx] = stolen_output
        index_to_category[idx] = category

    return indexed_outputs, index_to_category


def map_scores_to_categories(indexed_scores, index_to_category):
    """
    Converts:
        {1: 0.91, 2: 0.34}

    Into:
        {"Email": 0.91, "Code": 0.34}
    """
    return {
        index_to_category[int(idx)]: score
        for idx, score in indexed_scores.items()
    }


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
    target_output = fabricated_record["target_output"]

    indexed_stolen_outputs, index_to_category = build_indexed_stolen_outputs(
        records=records,
        model=model,
        best_input=best_input,
    )

    indexed_scores = llm.llm_based_outputs_comparison(
        target_output=target_output,
        stolen_outputs=indexed_stolen_outputs,
        model=args.evaluation_llm_model,
    )

    category_scores = map_scores_to_categories(
        indexed_scores=indexed_scores,
        index_to_category=index_to_category,
    )

    result = dict(sorted(category_scores.items(), key=lambda x: x[1], reverse=True))

    os.makedirs("model", exist_ok=True)

    out_path = args.out or f"model/diff_array_{args.theme}_{args.target_llm_model}{scenario_suffix}.json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved: {out_path}")