import numpy as np
import pandas as pd
from pn_model import param_space
from pn_model import eval_params

# meta-parameters: parallelized, output locale
parallel = True
tmp_path = "temp\\"
out_path = "outs\\"
out_file = "results"

# parameter names, min and max values, # steps
names = ['np', 'nr', 'npp', 'npr']
grids =  [[3, 200, 11], [2, 80, 9], [1, 13, 7], [1, 13, 7]]

# corresponding list of parameter combinations
nodes = param_space(*np.array(grids).transpose(), copies = 10,
                    scale = ['log', 'log', 'linear', 'linear'])

# run simulation on all parameter combinations
freqs, syncs, times, metas = eval_params(names, nodes, parallel, tmp_path)

# assemble results variables into a data frame
results = pd.DataFrame(np.hstack((nodes, times, metas, syncs, freqs)),
                       columns = names + ['time', 'dt', 'v0', 'sh', 'sso'] +
                       [a + ' ' + b for a in ['net', 'pace', 'relay']
                        for b in ['med', 'iqr', 'var', 'stable']])

# save combined data frame to csv file on disk
results.to_csv(out_path + out_file + ".csv")

# close main NEURON (if using ParallelContext)
if parallel:
    h.quit()