#!/bin/bash
for ((trial=33;trial<=50;trial++))
do
    echo $trial
    CUDA_VISIBLE_DEVICES=0 python cli.py train-model --trial=$trial
done
