 #! /bin/bash
 python3 train.py --verbose --use-cuda --dataset task --num-ways 5 --num-shots 5 --num-shots-test 5  --transform 'llvm'&&
 python3 train.py --verbose --use-cuda --dataset task --num-ways 5 --num-shots 5 --num-shots-test 5  --transform 'all'
