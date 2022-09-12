import numpy as np
from pn_model import PacemakerNetwork
from pn_model import simulate

# pn network parameters
npc = 86
nrc = 20
npp = 7
npr = 7
# simulation parameters
dt = 143.0296
v0 = -60.6886
sh = -10.2303

# generate pn network
pars = dict(zip(['np', 'nr', 'npp', 'npr'], [npc, nrc, npp, npr]))
net = PacemakerNetwork(pars)
# add trace recorders
net.add_voltage()
net.add_spikes(thresh = sh)

# run pn simulation
simulate(dt, v0)
# extract v_m traces
t = net.traces[0].time.as_numpy()
v = [[]] * len(net.traces)
for i in range(len(net.traces)):
    v[i] = net.traces[i].volt.as_numpy()
v = np.array(v).transpose()
# get freq statistics
sso, nf, fm, fi, fs = net.freq_calc(cell_info = True)
print(sso)

# concatenate outputs
fs = np.hstack((nf[[0, 2, 3], np.newaxis], np.vstack((fm, fi, fs))))
tv = np.concatenate((t[:, np.newaxis], v), axis = 1)
fstv = np.vstack((fs, tv))
# save to csv file
f = '_'.join([str(p) for p in [sso, npc, nrc, npp, npr]])
np.savetxt(f + '.csv', fstv, delimiter = ',')