#!/bin/bash

# Each item is the layer/a group of layers you want to unfreeze together
layer_groups=(
    "fc1 fc2"
    # "r1 fc1 fc2"
    # "fc2"
)

# Learning rates
lrs=(6e-4) # 1e-4, 2e-4, 4e-4 for screening

for layer_group in "${layer_groups[@]}"; do
    echo "Transfer learning with layers: $layer_group"

    # Convert the string "fc1 fc2" → array ["fc1", "fc2"]
    read -a layer_array <<< "$layer_group"

    for lr in "${lrs[@]}"; do
        python src/training/SSTL.py \
            --seed 5 \
            --unfreeze_layers "${layer_array[@]}" \
            --finetune_lr "$lr" \
            --finetuning_batch_size 1 \
            --finetune_epochs 5 \
            --thr 0.025
    done
done
