#!/bin/bash
#SBATCH -J GCN

if [ $1 == $"train" ]
	then python gcn_train.py train --num-epoch=$2 --save-freq=$3
elif [ $1 == $"resume" ]
	then python gcn_train.py resume $2 --num-epoch=$3 --save-freq=$4
elif [ $1 == $"inspect" ]
	then python gcn_train.py inspect $2
else
	echo "formats: "
	echo "1. train <num-epoch> <save-freq>"
	echo "2. resume <checkpoint> <num-epoch> <save-freq>"
	echo "3. inspect <checkpoint>"
fi