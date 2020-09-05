#! /bin/bash
python3 train.py --verbose --use-cuda  --num-ways 20 --num-shots 1 --num-shots-test 1 --dataset conv2d_graph_non-template --feature conv2d_graph_non-template --num-steps 1

python3 train.py --verbose --use-cuda  --num-ways 20 --num-shots 1 --num-shots-test 1 --dataset conv2d_graph_non-template --feature conv2d_graph_non-template --num-steps 5

python3 train.py --verbose --use-cuda  --num-ways 20 --num-shots 1 --num-shots-test 1 --dataset conv2d_2_graph_non-template --feature conv2d_2_graph_non-template --num-steps 1

python3 train.py --verbose --use-cuda  --num-ways 20 --num-shots 1 --num-shots-test 1 --dataset conv2d_2_graph_non-template --feature conv2d_2_graph_non-template --num-steps 5
