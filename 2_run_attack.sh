#!/bin/bash

# Run with: bash 2_run_attack.sh
# Purpose:
# 1) Run phase-2 prompt stealing/evaluation in parallel for all categories.
# 2) Write per-category logs and result/<category>_res_<target_model><scenario_suffix>.csv outputs.
# 3) Post-process results: merge into result/post_phase2_data_<target_model><scenario_suffix>.csv,
#    then delete per-category result CSVs.

set -euo pipefail

tasks="Ads Business Code Email Ideas SEO Writing Food Health Music Data Fashion Games Language Sports Study Translate Travel"

TARGET_LLM_MODEL="${TARGET_LLM_MODEL:-CommandA}"
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
> llm_calls.jsonl  # reset — count phase 2 calls only
STEAL_START=$(date +%s)

for task in $tasks
do
    nohup venv/bin/python -u 2_run_attack.py \
        --theme "$task" \
        $custom_arg \
        --target_llm_model "$TARGET_LLM_MODEL" \
        --out "${output_dir}/${task}_${TARGET_LLM_MODEL}${scenario_suffix}.txt" \
        >> "${output_dir}/${task}_${TARGET_LLM_MODEL}${scenario_suffix}.log" 2>&1 &

    current_jobs=$((current_jobs + 1))
    if [ "$current_jobs" -ge "$parallel_jobs" ]; then
        wait
        current_jobs=0
    fi
done

wait

STEAL_ELAPSED=$(( $(date +%s) - STEAL_START ))
STEAL_MIN=$(( STEAL_ELAPSED / 60 ))
STEAL_SEC=$(( STEAL_ELAPSED % 60 ))
mkdir -p result
echo "creating_stolen_prompts_and_pruning = ${STEAL_MIN}m ${STEAL_SEC}s" >> "result/final_documentation_${TARGET_LLM_MODEL}${scenario_suffix}.txt"

# Post-process per-category result files:
# 1) merge category results to a consolidated csv for next phase
# 2) delete each per-category result file
venv/bin/python3 - "$TARGET_LLM_MODEL" "$scenario_suffix" <<'PY'
import sys
import utils

target_model = sys.argv[1]
scenario_suffix = sys.argv[2]

merged_path = f"result/post_phase2_data_{target_model}{scenario_suffix}.csv"
utils.merge_phase2_results(merged_path, target_model, scenario_suffix)
PY

# LLM-calls counting — aggregate llm_calls.jsonl into final_documentation
venv/bin/python3 - "$TARGET_LLM_MODEL" "$scenario_suffix" <<'PY'
import sys, utils
utils.write_llm_call_summary(sys.argv[1], sys.argv[2], label="After Phase 2")
PY

echo "All attack jobs executed and post-processing completed."
