#!/bin/bash

for k in {0..9}
do
   python train.py $k
done
