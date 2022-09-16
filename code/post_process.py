import numpy as np
from pathlib import Path
import sys
import math

FNAME       = Path(sys.argv[1]).name 
DIR         = Path(sys.argv[1]).parent.as_posix() + '/'
OUTPUT_DIR  = DIR + FNAME[:-3] + '/outputs/'
TIME_CHUNK  = 50
SUBINT_CHUNK=4096
PERCENTILE 	= 90

scores = np.load(DIR + FNAME[:-3] + '/outputs/scores.npy')
aggregate_every = math.ceil(SUBINT_CHUNK/TIME_CHUNK)
scores_aggregated = np.add.reduceat(scores, np.arange(0, len(scores), aggregate_every))
percentile = np.percentile(scores_aggregated, PERCENTILE)
prediction = (scores_aggregated >= percentile)

data = np.hstack((prediction.reshape((-1, 1)), scores_aggregated.reshape((-1, 1))))
np.savetxt(DIR + FNAME[:-3] + '/outputs/predictions.csv', data, delimiter=',')



