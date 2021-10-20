#!/usr/bin/env bash

model_dir=$1
shift

model_path=$model_dir/model_multi

for dataset in $@; do
  echo $dataset
  for num_aug in 1 2 5; do
    pred_path=$model_dir/${dataset}_pred_aug${num_aug}.npz
    if [[ true || ! -f $pred_path ]]; then
      python -m inference_scripts.predict_$dataset --model-path="$model_path" --output-path="$pred_path" --num-aug=$num_aug
      python -m eval_scripts.eval_$dataset --pred-path="$pred_path" | tee "${pred_path}_eval.txt"
      echo ----
      echo
    fi
  done
done
