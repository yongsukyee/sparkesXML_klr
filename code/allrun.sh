#!/bin/bash
for filename in /datasets/work/mlaifsp-sparkes/work/sparkesX/multi/simplepulse/*.sf; do
    TESTVAR="${filename##*/}"
    sbatch run.sh $TESTVAR
done
