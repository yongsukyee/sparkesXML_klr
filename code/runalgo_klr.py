import pypsrfits
import evalmetricsimv0 as evalmetricsim
from klr.klr import Klr
from klr.helpers import SquaredExponential

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from pathlib import Path
from multiprocessing import Pool, set_start_method
from functools import partial
import time
import sys

###########################################################
import warnings
warnings.filterwarnings('ignore') #TODO: some library is throwing warnings when plotting. Not sure what or why
###########################################################

FNAME       = sys.argv[1] #'simplepulse_multi_01.sf'
DIR         = '/datasets/work/mlaifsp-sparkes/work/sparkesX/multi/simplepulse/'
OUTPUT_DIR  = DIR + FNAME[:-3] + '/outputs/'
TIME_CHUNK  = 50
VERBOSITY   = 0

def main(fname, verbosity=1):
    Path(OUTPUT_DIR).mkdir(parents = True, exist_ok=True)
    Path(OUTPUT_DIR + 'subints/').mkdir(parents = True, exist_ok=True)
    set_start_method('spawn') # https://pythonspeed.com/articles/python-multiprocessing/
    matplotlib_config()

    # Initialise KLR model
    klr = Klr(SquaredExponential(1.), precomputed_kernel=False, use_solver='scipy')
    # Load file
    psrfile = pypsrfits.PSRFITS(DIR + fname)
    # Get sim labels
    _, y_sim, sim, tframe = evalmetricsim.read_simlabel(DIR + fname, datadir='')
    print(y_sim)
    
    fout = open(OUTPUT_DIR + FNAME[:-3] + '.txt', 'w')
    fout.write('# subint t0 t1\n')
   
    #####
    # RUN ALGORITHM
    # Input bdata: 2d array with dimension [nfrequency, ntime]
    #####
    all_scores = []
    all_times = []
    for nrow in list(range(psrfile.nrows_file))[-3:]:
        compute_t0 = time.time()
        bdata, times, _ = psrfile.getData(nrow, None, get_ft=True, 
            squeeze=True, transpose=True)
        X, data = _make_X_data_pair(bdata, times)
        _plot_data_chunk(X, data, 'subints/' + str(nrow))
        t_indices = np.arange(0, bdata.shape[1], TIME_CHUNK)
        pool = Pool(processes=64)
        scores = pool.map(partial\
            (score_chunk, bdata, times, klr, nrow, verbosity), t_indices)
        pool.close()
        pool.join()
        print('Subint ' + str(nrow) + ' took ' + str(time.time() - compute_t0)\
            + ' seconds.')
        all_scores, all_times = update_scores_and_plot(all_scores, scores, 
            all_times, times)
        # Save to output if yes
        print(np.sum(scores))

        fout.write(f"{nrow} {times[0]} {times[-1]}\n")
    np.save(DIR + FNAME[:-3] + '/outputs/scores.npy', all_scores)
    fout.close()
    
def update_scores_and_plot(all_scores, scores, all_times, times):
    all_scores = np.hstack((all_scores, scores))
    all_times = np.hstack((all_times, np.linspace(times[0], times[-1], len(scores))))
    plt.plot(all_times, all_scores)
    plt.xlabel('Time (sec)')
    plt.ylabel('Score (??)')
    plt.savefig(OUTPUT_DIR + 'scores.png', bbox_inches='tight')
    plt.close()

    return all_scores, all_times

def score_chunk(bdata, times, klr, nrow, verbosity, t_idx):
    #print('Reading chunk ' + str(t_idx))
    last_idx = min(t_idx+TIME_CHUNK, bdata.shape[1])
    data_chunk = bdata[:, t_idx:last_idx]

    X, data_chunk = _make_X_data_pair(data_chunk, times[t_idx:last_idx])
    Path(OUTPUT_DIR + 'subints/' + str(nrow) + '/').\
        mkdir(parents = True, exist_ok=True)
    if verbosity > 1:
        _plot_data_chunk(X, data_chunk, 'subints/' + str(nrow) + '/' + str(t_idx))
    X_normalised = X.copy()
    X_normalised[:,1] = (X_normalised[:,1] - times[t_idx])/\
        (times[last_idx-1] -times[t_idx])

    klr.fit(X_normalised, data_chunk, num_iters=1, lamb=0.1)
    scores = klr.predict_proba(X_normalised)
    if verbosity > 0:
        _plot_data_chunk(X, scores, 'subints/' + str(nrow) + '/_' + str(t_idx))
    score = np.sum((data_chunk - 0.5) * (scores - 0.5))
    return score

def _make_X_data_pair(data, times = None):
    if not (times is None):
        t0 = times[0]; t1 = times[-1]
    else:
        t0 = 0; t1 = 1
    times01 = np.linspace(t0, t1, data.shape[1])
    freqs01 = np.linspace(1, 0, data.shape[0])
    X = np.transpose([np.tile(freqs01, len(times01)), np.repeat(times01, len(freqs01))])

    return X, np.reshape(data, (-1,1), order='F')

def _plot_data_chunk(X, data, fname='test'):
    matplotlib_config()
    plt.scatter(X[:,1], X[:,0], c=data, s=1, marker='s', vmin=0, vmax=1)
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (??)')
    plt.gca().ticklabel_format(useOffset=False)
    plt.savefig(OUTPUT_DIR + fname + '.png', bbox_inches = 'tight')
    plt.close()

def matplotlib_config():
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams['axes.labelsize'] = 30
    matplotlib.rcParams['legend.fontsize'] = 10
    matplotlib.rcParams['xtick.labelsize'] = 20
    matplotlib.rcParams['ytick.labelsize'] = 20
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"

if __name__ == '__main__':
    main(FNAME, verbosity = VERBOSITY)
