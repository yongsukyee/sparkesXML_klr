# SPARKESX with Machine Learning

## Data
Download the single beam data from [SPARKESX](https://doi.org/10.25919/fd4f-0g20) > Files > sbeam > all folders.

## Running the code
1. Install dependencies using ``python -m pip install -r requirements.txt``.
2. Depending on your setup, you may wish to run the code differently. 
3. We ran our expierments on a cluster that uses the SLURM job manager. To run code using SLURM, point to the location of the data on line 14 of ``run.sh``. Then us ``sbatch run.sh``.
4. If you want to manage individual python instances separately for each ``.sf`` file, you can run ``python runalgo_klr.py FNAME``, where FNAME is the absolute path to the ``.sf`` file.

## The outputs of the program
The verbosity of results that are saved is controlled by the VERBOSITY parameter on line 24 of ``runalgo_klr.py``.
1. If VERBOSITY <= 0. Results are written to a ``scores.npy`` file for each ``.sf`` file. These scores represent the power of each chunk as a function of time, and are also plotted in ``scores.png``.
2. If VERBOSITY <= 1. The function f of the RKHS (i.e. the logits of the bernoulli distribution) are plotted for each chunk of data. These are called ``_t.png'', where ``t`` is the chunk index. 
3. If VERBOSITY <= 2. The raw data itself is plotted in ``t.png``, where ``t`` is the chunk index.

## Collecting the outputs of the program
1. The file ``post_process.py`` and associated SLURM script ``post_process.sh`` put the scores in a ``predictions.csv`` file. This file associates to every time chunk a 0 or a 1, depending on whether an event is observed or not.
2. Given such post-processed ``predictions.csv`` files, ``collate_results.py`` produces the figures given in the paper. 
