#! /bin/bash
python3 train.py --verbose --use-cuda  --num-ways 20 --num-shots 1 --num-shots-test 1 --dataset graph_non-template --feature graph_non-template --num-steps 1
python3 train.py --verbose --use-cuda  --num-ways 20 --num-shots 1 --num-shots-test 1 --dataset graph_template --feature graph_template --num-steps 1
python3 train.py --verbose --use-cuda  --num-ways 20 --num-shots 1 --num-shots-test 1 --dataset curve --feature curve --num-steps 1

python3 train.py --verbose --use-cuda  --num-ways 20 --num-shots 1 --num-shots-test 1 --dataset graph_non-template --feature graph_non-template --num-steps 5
python3 train.py --verbose --use-cuda  --num-ways 20 --num-shots 1 --num-shots-test 1 --dataset graph_template --feature graph_template --num-steps 5
python3 train.py --verbose --use-cuda  --num-ways 20 --num-shots 1 --num-shots-test 1 --dataset curve --feature curve --num-steps 5
