#! /bin/bash

seeds=(42 43 44)

for seed in "${seeds[@]}"; do
    rm -rf data_cache/
    python graphany/run.py dataset=CoraXAll total_steps=500 n_hidden=64 n_mlp_layer=1 entropy=2 n_per_label_examples=5 seed=$seed use_wandb=true
    rm -rf data_cache/
    python graphany/run.py dataset=PubmedXAll total_steps=500 n_hidden=64 n_mlp_layer=1 entropy=2 n_per_label_examples=5 seed=$seed use_wandb=true
done
