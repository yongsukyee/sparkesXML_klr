#!/bin/bash
#for dir in simplepulse rfi unknown noise known combo real+combo; do
for dir in real+combo; do
    sbatch --job-name=$dir-multi post.sh
done
