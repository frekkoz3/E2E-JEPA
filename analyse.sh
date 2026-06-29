#!/bin/bash

set -e

model=$1

python3.11 -m src.utils.utils --csv models/$model/metrics.csv --save

mv models/$model/loss_plots/*.html imgs/

python3.11 -m src.validation.vis --config models/$model/config.yaml --weights models/$model/final.pkl

python3.11 -m src.validation.inspect --config models/$model/config.yaml --weights models/$model/final.pkl

python3.11 -m src.validation.attention --config models/$model/config.yaml --weights models/$model/final.pkl

python3.11 -m src.validation.multi_attention --config models/$model/config.yaml --weights models/$model/final.pkl