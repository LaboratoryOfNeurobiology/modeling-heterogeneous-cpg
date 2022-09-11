import numpy as np
from pn_model import PacemakerNetwork
from pn_model import simulate

npc = 86
nrc = 20
npp = 7
npr = 7

pars = dict(zip(['np', 'nr', 'npp', 'npr'], [npc, nrc, npp, npr]))
dt, v0, sh = 143.0296, -60.6886, -10.2303

net = PacemakerNetwork(pars)
net.add_voltage()
net.add_spikes(thresh = sh)
simulate(dt, v0)

t = net.traces[0].time.as_numpy()
v = [[]] * len(net.traces)
for i in range(len(net.traces)):
    v[i] = net.traces[i].volt.as_numpy()
v = np.array(v).transpose()

sso, nf, fm, fi, fs = net.freq_calc(cell_info = True)
print(sso)

fs = np.hstack((nf[[0, 2, 3], np.newaxis], np.vstack((fm, fi, fs))))
tv = np.concatenate((t[:, np.newaxis], v), axis = 1)
fstv = np.vstack((fs, tv))

f = '_'.join([str(p) for p in [sso, npc, nrc, npp, npr]])
np.savetxt(f + '.csv', fstv, delimiter = ',')