#!/bin/bash

for i in `seq 0 42`;
do
	nohup pipenv run python run_experiments.py --exp $i&
done

