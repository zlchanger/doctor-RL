#!/bin/bash

for MODEL_PATH in "$@"; do
    bash ragen/env/medical_consultation/evaluation/run_eval_patientllm_category.sh "$MODEL_PATH"
done