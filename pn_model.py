"""
Apteronotus leptorhynchus pacemaker nucleus NEURON model

Object-oriented library, including functionality for:
- construction of NEURON objects for pacemaker cells,
  relay cells, gap junctions, and voltage recorders
- construction of random pacemaker nucleus networks,
  calculation of their frequency metrics, and plotting
- calling NEURON to simulate neural activity of these
  networks for given (sets of) parameter combinations

i.ilies@northeastern.edu
last updated 2022-02-12
"""

from neuron import h
from abc import ABC
# arrays
import numpy as np
from numpy import random as rnd
from scipy import stats as sps
# graphs
import networkx as nx
from networkx.algorithms import bipartite as nxb
# plots
import matplotlib.pyplot as plt
# utils
import pandas as pd
import time

""" ELEMENTARY NEURON OBJECTS """

class Synapse(object):
    """ generates a gap junction between two NEURON sections at
    specified locations (default: end segment to first segment)
    and with specified conductance range (HalfGap defaults) """
    def __init__(self, pre_sec, post_sec, 
                 pre_x = 1, post_x = 0, gmin = 0.5, gmax = 10):
        self.name = pre_sec.name() + '_' + post_sec.name()
        # create HalfGap objects
        self.pre = h.HalfGap(pre_sec(pre_x))
        self.post = h.HalfGap(post_sec(post_x))
        # set cathode vs. anode
        self.pre.isanode = 1
        self.post.isanode = -1
        # set voltage pointers
        h.setpointer(post_sec(post_x)._ref_v, 'vgap', self.pre)
        h.setpointer(pre_sec(pre_x)._ref_v, 'vgap', self.post)
        # set conductance range
        self.pre.gmin = self.post.gmin = gmin
        self.pre.gmax = self.post.gmax = gmax

class Current(object):
    """ attaches current clamp to a NEURON section at specified
    location (default: first segment) with specified parameters
    (onset / delay, duration, and magnitude / amplitude)    """
    def __init__(self, sec, loc = 0, delay = 0, dur = 1, amp = 1):
        self.name = sec.name() + '_ie_' + str(int(loc * 100))
        # create IClamp object
        self.stim = h.IClamp(sec(loc))
        # set clamp parameters
        self.stim.delay = delay
        self.stim.dur = dur
        self.stim.amp = amp

class Voltage(object):
    """ attaches mebrane potential recorder to specified NEURON
    section at specified location; also records time stamps """
    def __init__(self, sec, loc = 1):
        self.name = sec.name() + '_vm_' + str(int(loc * 100))
        # membrane potential recorder
        self.volt = h.Vector()
        self.volt.record(sec(loc)._ref_v)
        # corresponding time recorder
        self.time = h.Vector()
        self.time.record(h._ref_t)

class Spikes(object):
    """ attaches action potential recorder to specified NEURON
    section at specified location with specified threshold """
    def __init__(self, sec, loc = 1, thresh = 0):
        self.name = sec.name() + '_ap_' + str(int(loc * 100))
        # spike (action potential) counter
        self.apc = h.APCount(sec(loc))
        self.apc.thresh = thresh
        # corresponding spike time recorder
        self.spikes = h.Vector()
        self.apc.record(self.spikes)

""" PACEMAKER NUCLEUS NEURONS """

class NetworkCell(ABC):
    """ simple two-section neuron (soma and axon, no dendrites)
    with active Hodgekin-Huxley channels for both sections  """
    
    def __init__(self, ID, sub_type, pars = None):
        """ create a new neuron with specified identifier, type
        (defined by sub-class), and geometrical and biophysical
        parameters (provided as dictionary; default: None)  """
        self.type = sub_type
        self.name = self.type + '_' + str(ID)
        # create cell sections
        self.soma = h.Section(name = self.name + '_soma')
        self.axon = h.Section(name = self.name + '_axon')
        # connect cell sections
        self.axon.connect(self.soma(1))
        # update any parameters
        self.set_geom(pars)
        self.set_soma(pars)
        self.set_axon(pars)
        # construct cell in 3D
        self.shape_3D()
    
    """BUILDING FUNCTIONS"""
    @staticmethod
    def set_geometry(sec, L, D, N):
        """ set geometry attributes for input NEURON section """
        sec.L = L       # section length (um)
        sec.diam = D    # section diameter (um)
        sec.nseg = N    # number of segments
    
    @staticmethod
    def set_biophysics(sec, Ra, Cm, gNa, gK, gL, ENa, EK, EL):
        """ assign membrane properties across NEURON section """
        sec.Ra = Ra     # axial resistance (Ohm*cm)
        sec.cm = Cm     # membrane capacitance (uF/cm2)
        # insert active Hodgkin-Huxley currents (Na, K, leak)
        sec.insert('hh')
        sec.insert('na_ion')
        sec.insert('k_ion')
        # set conductances and equlibrium/reversal potentials
        for seg in sec:
            seg.hh.gnabar = gNa     # max sodium conductance (S/cm2)
            seg.ena = ENa           # sodium equilibrium potential (mV)
            seg.hh.gkbar = gK       # max potassium conductance
            seg.ek = EK             # potassium equilibrium potential
            seg.hh.gl = gL          # leak conductance
            seg.hh.el = EL          # leak reversal potential

    def set_geom(self, sD, sdx, aL, aD, adx):
        """ assign geometry parameters to cell soma and axon """
        self.set_geometry(self.soma, sD, sD, int(sD / sdx))
        self.set_geometry(self.axon, aL, aD, int(aL / adx))
    
    def set_soma(self, Ra, Cm, gNa, gK, gL, ENa, EK, EL):
        """ assign membrane properties to all soma segments """
        self.set_biophysics(self.soma, Ra, Cm, gNa, gK, gL, ENa, EK, EL)

    def set_axon(self, Ra, Cm, gNa, gK, gL, ENa, EK, EL):
        """ assign membrane properties to all axon segments """
        self.set_biophysics(self.axon, Ra, Cm, gNa, gK, gL, ENa, EK, EL)
    
    """GEOMETRY FUNCTIONS"""
    def shape_3D(self):
        """ set the default shape of the cell in 3D space; set
        soma(0) to origin and axon extending along X-axis  """
        # extract current lengths (will be destroyed)
        sL = self.soma.L
        aL = self.axon.L
        # delete current 3D coordinates (if existing)
        h.pt3dclear(sec=self.soma)
        h.pt3dclear(sec=self.axon)
        # place soma from 0 along x-axis
        h.pt3dadd(0, 0, 0, self.soma.diam, sec = self.soma)
        h.pt3dadd(sL, 0, 0, self.soma.diam, sec = self.soma)
        # place axon extending from soma
        h.pt3dadd(sL, 0, 0, self.axon.diam, sec=self.axon)
        h.pt3dadd(sL + aL, 0, 0, self.axon.diam, sec=self.axon)
        # reset cell position attributes
        self.x = self.y = self.z = 0

    def set_position(self, x, y, z):
        """ shift base (soma(0)) location to new 3D coordinates
        then move all cell parts relative to that location  """
        # shift each section by dx, dy, dz
        for sec in [self.soma, self.axon]:
            for i in range(sec.n3d()):
                h.pt3dchange(i, x - self.x + sec.x3d(i),
                             y - self.y + sec.y3d(i),
                             z - self.z + sec.z3d(i),
                             sec.diam3d(i), sec = sec)
        # update cell position attributes
        self.x, self.y, self.z = x, y, z

    def rotateZ(self, theta):
        """ rotate cell about the Z-axis by the specified angle
        while keeping the base location (soma(0)) fixed     """
        # rotate each 3D point in each section within XY-plane
        for sec in [self.soma, self.axon]:
            for i in range(sec.n3d()):
                dx = (self.x + np.cos(theta) * (sec.x3d(i) - self.x) -
                      np.sin(theta) * (sec.y3d(i) - self.y))
                dy = (self.y + np.sin(theta) * (sec.x3d(i) - self.x) +
                      np.cos(theta) * (sec.y3d(i) - self.y))
                h.pt3dchange(i, dx, dy, 0, sec.diam3d(i), sec = sec)

    """UTILITY FUNCTIONS"""
    def add_synapse(self, target, gmin = 0.5, gmax = 10):
        """ connects midpoint of last axonal segment of current
        cell to midpoint of first somatic segment of target one
        through a gap junction with specified conductances """
        return Synapse(self.axon, target.soma, 1 - 1 / self.axon.nseg / 2, 
                       1 / target.soma.nseg / 2, gmin, gmax)

    def add_current(self, sec = "soma", delay = 0, dur = 1, amp = 1):
        """ attaches current clamp to midpoint of first segment
        of specified section; stimulates this cell with a pulse
        of current with specified delay (ms), duration (ms) and
        amplitude (nA)                                      """
        sec = getattr(self, sec)
        return Current(sec, 1 / sec.nseg / 2, delay, dur, amp)

    def get_voltage(self, sec = "soma"):
        """ attaches a membrane potential recording process to
        midpoint of last segment of specified section      """
        sec = getattr(self, sec)
        return Voltage(sec, 1 - 1 / sec.nseg / 2)
        
    def get_spikes(self, sec = "soma", thresh = 0):
        """ attaches action potential detector and recorder to
        midpoint of last segment of specified section and with
        specified activation threshold (mV)                """
        sec = getattr(self, sec)
        return Spikes(sec, 1 - 1 / sec.nseg / 2, thresh)


class PacemakerCell(NetworkCell):
    """ sub-class for pacemaker cell; sets cell type to 'pace'
    and provides default values for geometry and biophysics """
    
    def __init__(self, ID = 0, pars = None):
        super().__init__(ID, 'pace', pars)
    
    def set_geom(self, pars = None):
        super().set_geom(*rep_pars({'psD': 32, 'psdx': 1, 'paL': 42,
                                    'paD': 5, 'padx': 1}, pars).values())
    
    def set_soma(self, pars = None):
        super().set_soma(*rep_pars({'psRa': 100., 'psCm': 1., 'psgNa': 1.3,
                                    'psgK': 0.16, 'psgL': 0.0001, 'ENa': 61.,
                                    'EK': -61., 'EL': -70.}, pars).values())
    
    def set_axon(self, pars = None):
        super().set_axon(*rep_pars({'paRa': 100., 'paCm': 1., 'pagNa': 0.5,
                                    'pagK': 0.02, 'pagL': 0.001,'ENa': 61.,
                                    'EK': -61., 'EL': -70.}, pars).values())


class RelayCell(NetworkCell):
    """ sub-class for relay cell; sets cell type to 'relay' and
    provides default values for geometry and biophysics     """
    
    def __init__(self, ID = 0, pars = None):
        super().__init__(ID, 'relay', pars)
    
    def set_geom(self, pars = None):
        super().set_geom(*rep_pars({'rsD': 76, 'rsdx': 1, 'raL': 35, 
                                    'raD': 6, 'radx': 1}, pars).values())
    
    def set_soma(self, pars = None):
        super().set_soma(*rep_pars({'rsRa': 100., 'rsCm': 1., 'rsgNa': 0.9,
                                    'rsgK': 0.16,'rsgL': 0.0003, 'ENa': 61.,
                                    'EK': -61., 'EL': -70.}, pars).values())
    
    def set_axon(self, pars = None):
        super().set_axon(*rep_pars({'raRa': 500., 'raCm': 1., 'ragNa': 0.5,
                                    'ragK': 0.05, 'ragL': 0.001, 'ENa': 61.,
                                    'EK': -61., 'EL': -70.}, pars).values())

""" PACEMAKER NUCLEUS NETWORK """

class PacemakerNetwork():
    """ directed network comprised of pacemaker and relay cells
    with specified pacemaker-to-pacemaker, pacemaker-to-relay,
    and relay-to-relay gap-junction connectivity patterns; also
    includes supporting and monitoring point processes (current
    injection, membrane potential recorder, spike recorder) """
    
    def __init__(self, pars = None):
        """ generate NetworkX graph and core NEURON objects """
        self.graph = []     # connectivity patterns (NetworkX graph)
        self.cells = []     # list of cells (linked NEURON sections)
        self.types = []     # list of cell types (pace: 1, relay: -1)
        self.synapses = []  # list of synapses (NEURON halfgap pairs)
        # generate topology
        self.set_topology(pars)
        # construct network
        self.build_network(pars)
        # initialize others
        self.spikes = []    # list of spike tracers (NEURON objects)
        self.inputs = []    # list of current inputs (NEURON clamps)
        self.traces = []    # list of voltage traces (NEURON arrays)
    
    def __del__(self):
        """ deletes NEURON sections associated with network """
        # clean up NEURON workspace
        for cell in self.cells:
            h.delete_section(sec = cell.soma)
            h.delete_section(sec = cell.axon)
    
    def set_topology(self, pars = None):
        """generates directed bipartite-like graph of specified
        size and connectivity; design parameters different from
        default values (node counts, and average and dispersion
        of in- and out-degrees) expected as a dictionary    """
        # default parameter values
        defs = {'np': 87,   # total number of pacemaker cells
                'nr': 20,   # total number of relay cells
                'npp': 6,   # average number of pacemaker-pacemaker connections
                'ppo': 1,   # dispersion in outgoing pacemaker-pacemaker connections
                'ppi': 1,   # dispersion in incoming pacemaker-pacemaker connections
                'npr': 7,   # average number of pacemaker-relay connections
                'pro': 1,   # dispersion in outgoing pacemaker-relay connections
                'pri': 1,   # dispersion in incoming pacemaker-relay connections
                'nrr': 0,   # average number of relay-relay connections
                'rro': 1,   # dispersion in outgoing relay-relay connections
                'rri': 1}   # dispersion in incoming relay-relay connections
        # replace with user values
        defs = rep_pars(defs, pars)
        # start with pacemaker-relay connections
        self.graph = self.mynet(defs['np'], defs['nr'], defs['np'] *
                                defs['npr'], defs['pro'], defs['pri'])
        # append pacemaker-pacemaker connections
        g = self.mynet(defs['np'], 0, defs['np'] * defs['npp'],
                            defs['ppo'], defs['ppi'])
        self.graph.add_edges_from(g.edges)
        # append relay-relay connections, if any
        g = self.mynet(defs['nr'], 0, defs['nr'] * defs['nrr'],
                             defs['rro'], defs['rri'])
        self.graph.add_edges_from([(i + defs['np'], j + defs['np']) for (i, j) in g.edges])
        # save cell type as numeric value (+/-1)
        self.types = ([1] * int(defs['np'])) + ([-1] * int(defs['nr']))
    
    def build_network(self, pars = None):
        """ creates sets of pacemaker and relay cells and links
        them via gap junctions according to graph structure """
        # create cells
        for i in range(len(self.graph)):
            if self.types[i]> 0:
                self.cells.append(PacemakerCell(i, pars))
            else:
                self.cells.append(RelayCell(i, pars))
        # add synapses
        for (i, j) in self.graph.edges():
            self.synapses.append(self.cells[i].add_synapse(self.cells[j]))
    
    @staticmethod
    def mynet(no, ni, k, do, di):
        """ generates random directed graph (regular/bipartite)
        with specified numbers of nodes (no, optionally ni) and
        edges (k) and dispersion parameters for out-degrees and
        in-degrees (do, di) distributions                   """
        no, ni, k = int(no), int(ni), int(k)
        # bipartite graph if ni argument value specified and positive
        if ni is not None and ni > 0:
            bi = True
        else:
            bi = False
            ni = no
        if di == 1 and do == 1:
            # degrees are Poisson distributed; random directed graphs
            if bi and k >= no * ni:
                g = nxb.random_graph(no, ni, 0, directed = True)
                g.add_edges_from([(i, j) for i in range(no)
                                  for j in range(no, no + ni)])
            elif bi:
                g = nxb.gnmk_random_graph(no, ni, k, directed = True)
            else:
                g = nx.gnm_random_graph(no, k, directed = True)
        else:
            # randomly generate sequence of out-degrees
            out_deg = PacemakerNetwork.myrand(no, k, do)
            # randomly generate sequence of in-degrees
            in_deg = PacemakerNetwork.myrand(ni, k, do)
            # use configuration model with these degree distributions
            if bi:
                g = nxb.alternating_havel_hakimi_graph(out_deg, in_deg)
                ### NOTE GRAPH MAY HAVE MULTI-EDGES ###
            else:
                g = nx.directed_havel_hakimi_graph(in_deg, out_deg)
        return g
    
    @staticmethod
    def myrand(k, s, d):
        """ randomly generates k integers with sum s (mean s/k)
        and dispersion measure d, using the following methods:
            d < 0: uniform with range 2*abs(d) (non-negative)
            d = 0: constant (or within +/-1 if s/k not integer)
            d = 1: Poisson with average = variance = s/k
            d > 1: negative binomial with variance d*s/k
            d in (0,1): binomial with variance d*s/k        """
        if k == 0:
            # special case -- nothing to do, return
            return []
        elif s == 0:
            # special case -- nothing to randomize
            return [0] * k
        # compute mean, variance
        mu, va = s / k, d * s / k
        if d == 0:
            # ~equal values -- use numpy range to generate cumulative
            # sums, then use numpy diff to extract the actual values
            x = np.diff(np.arange(0, s + mu, mu).astype(int))
            rnd.shuffle(x)  # to avoid periodic values if s/k non-int
        elif d < 0:
            # uniform integers within +/-floor(d) of the mean and > 0
            x = rnd.randint(np.ceil(mu + d), np.floor(mu - d) + 1, k)
        elif np.isclose(mu, va):
            # poisson with lambda = mu
            x = rnd.poisson(mu, k)
        elif va < mu:
            # binomial with p = 1 - var / mu, n = mu / p 
            x = rnd.binomial(mu ** 2 / (mu - va), 1 - va / mu, k)
        elif va > mu:
            # negative binomial with p = mu / var, n = mu (1 - p) / p
            x = rnd.negative_binomial(mu ** 2 / (va - mu), mu / va, k)
        # ensure sum equals s -- need exact match for graph generator
        # randomly select entries with replacement, then increment or
        # decrement their values by 1 until the target sum is reached
        c = np.sign(s - sum(x))
        while sum(x) != s:
            # to prevent variance inflation, weights are proportional
            # to values when the sum is being decreased, respectively 
            # inversely proportional to values when it is increased;
            # special case -- equal weights when all values identical
            if min(x) == max(x):
                p = x + 1
            elif c > 0:
                p = max(x) - x
            else:
                p = x - min(x)
            j = rnd.choice(range(k), p = p / max(1, sum(p)))
            x[j] += c
        return x
    
    def add_current(self, sec = "soma", delay = 0, dur = 1, amp = 1):
        """ attach current clamps with specified parameters to
        to first segment of specified section of each cell """
        for cell in self.cells:
            self.inputs.append(cell.add_current(sec, delay, dur, amp))
    
    def add_voltage(self, sec = "soma"):
        """ attach membrane potential recorders to last segment
        of specified section of each network cell           """
        for cell in self.cells:
            self.traces.append(cell.get_voltage(sec))
    
    def add_spikes(self, sec = "soma", thresh = 10):
        """ attach action potential detectors/recorders to last
        segment of specified section of each network cell   """
        for cell in self.cells:
            self.spikes.append(cell.get_spikes(sec, thresh))
    
    def freq_calc(self, pop_info = False, cell_info = False):
        """ calculates several network-level, population-level,
        and cell-level firing frequency measurements (medians,
        IQRs, stability); returns only network-level statistics
        if not specified otherwise; first output = boolean flag
        for synchronized sustained oscillations             """
        # return None if actions potentials not tracked
        if not self.spikes:
            return None
        # allocate cell summaries (median, IQR, stable)
        f_med = np.zeros(len(self.cells))
        f_iqr = np.zeros(len(self.cells))
        f_sta = np.zeros(len(self.cells), dtype = bool)
        for i in range(len(self.cells)):
            # extract list of spikes
            t_spikes = self.spikes[i].spikes.as_numpy()
            # skip if too few spikes
            if len(t_spikes) < 2:
                continue
            # compute instantaneous frequencies as the
            # inverses of observed inter-spike intervals
            dt_spikes = np.diff(t_spikes)
            inst_freq = 1000 / dt_spikes
            # calculate median and IQR (non-parametric
            # to adjust for outliers and multimodality)
            f_med[i] = np.median(inst_freq)
            f_iqr[i] = sps.iqr(inst_freq)
            # linear regression on frequency estimates
            # excluding initial third of simulation to
            # avoid contamination from initialization
            idx = np.flatnonzero(t_spikes > h.tstop / 3)
            if len(idx) > 3:
                *_, p, e = sps.linregress(t_spikes[idx[1:]],
                                          inst_freq[idx[:-1]])
            else:
                p = 0
            # set stable flag to 1 if last spike timing
            # within [median + 2 IQR] of simulation end
            # (else assume decay) and if frequency slope
            # not different from 0 (else assume drift)
            if p > 0.01 and ((h.tstop - t_spikes[-1]) < 1000 /
                            (f_med[i] - 2 * f_iqr[i])):
                f_sta[i] = 1
        # network summaries (median and IQR of medians,
        # median of intra-cell IQRs, proportion stable)
        f_net = np.array([np.median(f_med), sps.iqr(f_med),
                          np.median(f_iqr), np.mean(f_sta)])
        # label as sustained synchronized oscillations
        # if cells stable and with similar frequencies
        if np.all(f_sta) and np.isclose(np.min(f_med), np.max(f_med), atol = 1):
            f_sso = 1
        else:
            f_sso = 0
        # return network summaries and optionally cell
        # details if no population summaries requested
        if not pop_info and not cell_info:
            return f_sso, f_net
        elif not pop_info:
            return f_sso, f_net, f_med, f_iqr, f_sta
        # compute similar summaries at population level
        isp = np.array(self.types) > 0
        f_pace = [np.median(f_med[isp]), sps.iqr(f_med[isp]),
                  np.median(f_iqr[isp]), np.mean(f_sta[isp])]
        isr = np.array(self.types) < 0
        f_relay = [np.median(f_med[isr]), sps.iqr(f_med[isr]),
                   np.median(f_iqr[isr]), np.mean(f_sta[isr])]
        # return network and population summaries, as
        # well as individual cell details if requested
        if not cell_info:
            return f_sso, f_net, f_pace, f_relay
        else:
            return f_sso, f_net, f_pace, f_relay, f_med, f_iqr, f_sta
    
    def plot_raster(self, save_loc = '', save_tag = 'test'):
        """ raster plot of spike times across cell indices """
        plt.switch_backend('agg')
        # define default line colors
        line_cols = {'pace': {'soma': 'black', 'axon': 'red'},
                      'relay': {'soma': 'blue', 'axon': 'green'}}
        # generate and label 2D axes
        ax = plt.axes()
        ax.set_xlabel('Simulation Time (ms)')
        ax.set_ylabel('Cell Identifier (#)')
        ax.set_title(str(save_tag))
        # loop through all recorders
        for spike in self.spikes:
            # parse object identifier
            cel, idx, sec, *_ = spike.name.split('_')
            # mark spikes with lines
            plt.vlines(spike.spikes, int(idx) + 0.3, int(idx) + 1.3, 
                          colors = line_cols[cel][sec], alpha = 0.6)
        # save figure to disk as pdf
        if len(save_loc) == 0:
            plt.savefig(f"{save_tag} raster.pdf", format = 'pdf')
        else:
            plt.savefig(f"{save_loc}/{save_tag} raster.pdf", format = 'pdf')
        plt.close()
    
    def plot_traces(self, save_loc = '', save_tag = 'test'):
        """ mebrane potential plots over time across cells """
        plt.switch_backend('agg')
        # define default line colors
        line_cols = {'pace': {'soma': 'black', 'axon': 'red'},
                      'relay': {'soma': 'blue', 'axon': 'green'}}
        # generate and label 3D axes
        ax = plt.axes(projection = '3d')
        ax.set_xlabel('Simulation Time (ms)')
        ax.set_ylabel('Cell Identifier (#)')
        ax.set_zlabel('Membrane Potential (mV)')
        ax.set_title(str(save_tag))
        # loop through all recorders
        for trace in self.traces:
            # parse object identifier
            cel, idx, sec, *_ = trace.name.split('_')
            # extract the time stamps
            t = trace.time
            # plot membrane potential
            ax.plot3D(t, int(idx) * np.ones(len(t)), trace.volt, 
                      c = line_cols[cel][sec], alpha = .4)
        # save figure to disk as pdf
        if len(save_loc) == 0:
            plt.savefig(f"{save_tag} traces.pdf", format = 'pdf')
        else:
            plt.savefig(f"{save_loc}/{save_tag} traces.pdf", format = 'pdf')
        plt.close()

""" NETWORK STUDY FUNCTIONS """

def network_func(pars = None, dt = 100, ns = 100, sh = 20):
    """ generates a Pn network with a given set of parameters,
    simulates it for the specified period, and returns various
    frequency measures and internal simulation parameters  """
    # run test model
    dt, v0, sh = network_test(pars, dt, ns, sh)
    # build NEURON model
    net = PacemakerNetwork(pars)
    net.add_spikes(thresh = sh)
    # run Pn simulation
    simulate(dt, v0)
    # compute frequency
    sso, f_net, f_pace, f_relay = net.freq_calc(pop_info = True)
    # aggregate outputs
    return *f_net, *f_pace, *f_relay, sso, dt, v0, sh

def network_test(pars = None, dt = 100, ns = 100, sh = 20):
    """ test run using minimal 2-cell system (1 pacemaker and 1
    relay, connected) for specified duration; used to estimate
    the duration needed to observe the desired number of spikes
    (if any possible), optimal action potential threshold (for
    specified minimum spike height), and resting potential  """
    # create test network with revised parameters
    test = PacemakerNetwork(rep_pars(pars.copy(),
                            {'np': 1, 'nr': 1, 'npp': 0, 'npr': 1}))
    # add membrane potential and spike recorders
    test.add_voltage()
    test.add_spikes()
    # simulate 2-cell system for provided period
    simulate(dt)
    # estimate resting potential = minimum value
    # across membrane potential traces after the
    # first third of simulation period (to avoid
    # initialization / synchronization artifacts)
    idx = test.traces[0].time.as_numpy() > dt / 3
    vmp = test.traces[0].volt.as_numpy()
    vmr = test.traces[1].volt.as_numpy()
    vmin = np.array([np.min(vmp[idx]), np.min(vmr[idx])])
    # identify action potential threshold = mid-
    # point between minimum and maximum membrane
    # potential values, if differences are large
    # enough for either pacemaker or relay cells
    vmax = np.array([np.max(vmp[idx]), np.max(vmr[idx])])
    if np.min(vmax - vmin) > sh:
        thresh = np.mean(np.hstack([vmin, vmax]))
    elif np.max(vmax - vmin) > sh:
        thresh = np.mean(np.vstack([vmin, vmax])[:, np.argmax(vmax - vmin)])
    else:
        thresh = 0
    # estimate simulation duration = inverse of
    # lowest frequency if positive, of highest
    # frequency otherwise, or None if both == 0
    # (n = 2: min = med - IQR, max = med + IQR)
    *_, f_net = test.freq_calc()
    if f_net[0] > f_net[1]:
        dt = ns / (f_net[0] - f_net[1]) * 1000
    elif f_net[0] > 0:
        dt = ns / (f_net[0] + f_net[1]) * 1000
    #
    return dt, np.mean(vmin), thresh

def network_wrap(pars = None, idx = None, temp_path = None):
    """ generates a Pn network with a given set of parameters;
    returns various frequency measures, internally determined
    simulation parameters, run time, and optional job ID  """
    t0 = time.time()
    freqs = []
    # run Pn simulation
    *freqs, sso, dt, v0, sh = network_func(pars)
    # print run summary
    print("run", idx, ":", round(time.time() - t0), "s,",
          round(freqs[0]), "Hz,", sso == 1)   
    # save to temp file
    if temp_path is not None:
        np.savetxt(temp_path + str(int(time.time() * 1e7)) + ".csv",
                   [idx, sso] + freqs)
    # aggregate outputs
    return freqs, sso, (dt, v0, sh), (time.time() - t0), idx

""" PARAMETER SPACE FUNCTIONS """

def param_space(lower, upper, steps, copies = 1, scale = 'linear'):
    """ generates a list of parameter combinations given lower
    and upper bounds, and number of steps, for each considered
    parameter; arguments must have the same length; degenerate
    inputs (steps = 1) accepted; returns parameter matrix with
    one row per combination and one column per parameter """
    # construct 1-dimensional grids
    # all linear, all logarithmic
    if scale == 'linear':
        grids = [np.unique(np.linspace(lower[j], upper[j], steps[j],
                    dtype = int)) for j in range(len(lower))]
    elif scale == 'log':
        grids = [np.unique(np.geomspace(lower[j], upper[j], steps[j],
                    dtype = int)) for j in range(len(lower))]
    # or each paramter different
    else:
        grids = []
        for j in range(len(lower)):
            if scale[j] == 'linear':
                grids.append(np.unique(np.linspace(lower[j], upper[j],
                                                   steps[j], dtype = int)))
            elif scale[j] == 'log':
                grids.append(np.unique(np.geomspace(lower[j], upper[j],
                                                    steps[j], dtype = int)))
    # construct n-dimensional grids
    nodes = np.meshgrid(*grids)
    # reshape individual components
    for j in range(len(nodes)):
        nodes[j] = np.ravel(nodes[j])
    # aggregate into regular matrix
    nodes = np.column_stack(nodes)
    # append replicates if required
    if copies > 1:
        nodes = np.tile(nodes, (copies, 1))
    #
    return nodes

def eval_params(names, nodes, use_mpi = True, temp_path = None):
    """ simulates Pn network for list of parameter combinations
    given as list of names and table of corresponding parameter
    values per combination, and returns corresponding frequency
    measurements (median, etc.) and run times w/ or w/o MPI """
    # allocate outputs: frequency, others
    freqs = [[]] * len(nodes)
    syncs = np.zeros(len(nodes))
    times = np.zeros(len(nodes))
    metas = [[]] * len(nodes)
    # initialize ParallelContext with MPI
    if use_mpi:
        h.nrnmpi_init()
        pc = h.ParallelContext()
        pc.runworker()
        # submit all combinations as jobs
        for idx, node in enumerate(nodes):
            pc.submit(network_wrap, dict(zip(names, node)), idx, temp_path)
        # collect outputs from MPI workers
        while pc.working():
            freq, sso, meta, dtim, idx = pc.pyret()
            idx = int(idx)
            freqs[idx] = freq
            syncs[idx] = sso
            times[idx] = dtim
            metas[idx] = meta
        # close ParallelContext
        pc.done()
    # simple loop over all combinations
    else:
        for idx, node in enumerate(nodes):
            freqs[idx], syncs[idx], metas[idx], times[idx], _ = \
                network_wrap(dict(zip(names, node)), idx, temp_path)
    # convert all outputs to 2d arrays
    return np.array(freqs), syncs[:, np.newaxis], times[:, np.newaxis], np.array(metas)

""" OTHER / UTILITY FUNCTIONS """

def rep_pars(def_vals, user_vals):
    """ replaces default parameter values (given as dictionary)
    with user provided ones (also dictionary) if keys match """
    if user_vals is not None:
        for k in user_vals.keys():
            if k in def_vals.keys():
                def_vals[k] = user_vals[k]
    return def_vals

def simulate(tstop, vini = -60, temp = 27):
    """ initialize and run NEURON simulation for the specified
    duration, using adaptive integration and specified initial
    membrane potential value and temperature               """
    h.load_file("stdrun.hoc")
    h.cvode_active(1)
    h.v_init = vini
    h.celsius = temp
    h.tstop = tstop
    h.run()
