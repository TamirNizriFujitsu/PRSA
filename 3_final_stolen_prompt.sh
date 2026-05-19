#!/bin/bash

# Run with: bash 3_final_stolen_prompt.sh
# Purpose:
# 1) Run phase-3 category ranking in parallel for all categories.
# 2) Write per-category model/diff_array_<category>_<target_model><scenario_suffix>.json.
# 3) Merge all per-category diff arrays into model/diff_array_all_<target_model><scenario_suffix>.json
#    (per-category files are kept).

set -euo pipefail

echo "phase 3 started"

tasks="Ads Business Code Email Ideas SEO Writing Food Health Music Data Fashion Games Language Sports Study Translate Travel"

TARGET_LLM_MODEL="${TARGET_LLM_MODEL:-gpt-4o}"
CUSTOM="${CUSTOM:-false}"
CUSTOM_SCENARIO="${CUSTOM_SCENARIO:-custom}"
scenario_suffix=""
custom_arg=""
if [ "$CUSTOM" = "true" ]; then
    scenario_suffix="_${CUSTOM_SCENARIO}"
    custom_arg="--custom --custom_scenario ${CUSTOM_SCENARIO}"
fi

parallel_jobs=6
output_dir="log"
mkdir -p "$output_dir"

current_jobs=0
> llm_calls.jsonl  # reset — count phase 3 calls only
mkdir -p result

DIFF_ARRAYS_START=$(date +%s)

for task in $tasks
do
    nohup venv/bin/python -u 3_final_stolen_prompt.py \
        --theme "$task" \
        $custom_arg \
        --target_llm_model "$TARGET_LLM_MODEL" \
        >> "${output_dir}/${task}_phase3_${TARGET_LLM_MODEL}${scenario_suffix}.log" 2>&1 &

    current_jobs=$((current_jobs + 1))
    if [ "$current_jobs" -ge "$parallel_jobs" ]; then
        wait
        current_jobs=0
    fi
done

wait

DIFF_ARRAYS_ELAPSED=$(( $(date +%s) - DIFF_ARRAYS_START ))
DIFF_ARRAYS_MIN=$(( DIFF_ARRAYS_ELAPSED / 60 ))
DIFF_ARRAYS_SEC=$(( DIFF_ARRAYS_ELAPSED % 60 ))
echo "Phase 3.1 - creating_diff_arrays_for_all_categories = ${DIFF_ARRAYS_MIN}m ${DIFF_ARRAYS_SEC}s" >> "result/final_documentation_${TARGET_LLM_MODEL}${scenario_suffix}.txt"

# Merge all per-category diff arrays into one combined JSON and delete per-category files.
# File format: {<category>: {<category_1>: diff,...., <category_18>: diff}}
venv/bin/python3 - "$TARGET_LLM_MODEL" "$scenario_suffix" <<'PY'
import sys
import utils

target_model = sys.argv[1]
scenario_suffix = sys.argv[2]

out_path = f"result/diff_array_all_{target_model}{scenario_suffix}.json"
utils.merge_phase3_diff_arrays(out_path, target_model, scenario_suffix)
PY

ASR_START=$(date +%s)

venv/bin/python3 - "$TARGET_LLM_MODEL" "$scenario_suffix" <<'PY'
import sys
import utils
import scorers

target_model = sys.argv[1]
scenario_suffix = sys.argv[2]

best_category = utils.choose_best_stolen_prompt(target_model, scenario_suffix)
print(f"Best stolen prompt category: {best_category}")

# best_stolen_prompt, asr = scorers.calculate_asr(best_category, target_model, scenario_suffix)
# print(f"Best stolen prompt: {best_stolen_prompt}")
# print(f"ASR: {asr}")
PY

ASR_ELAPSED=$(( $(date +%s) - ASR_START ))
ASR_MIN=$(( ASR_ELAPSED / 60 ))
ASR_SEC=$(( ASR_ELAPSED % 60 ))
echo "phase 3.2 - calculating_stolen_prompts_asr = ${ASR_MIN}m ${ASR_SEC}s" >> "result/final_documentation_${TARGET_LLM_MODEL}${scenario_suffix}.txt"

# Full pipeline LLM call summary (phases 1+2+3)
venv/bin/python3 - "$TARGET_LLM_MODEL" "$scenario_suffix" <<'PY'
import sys, utils
utils.write_llm_call_summary(sys.argv[1], sys.argv[2], label="After phase 3")
PY

echo "All phase-3 jobs executed and diff arrays merged."

echo "phase 3 ended"
