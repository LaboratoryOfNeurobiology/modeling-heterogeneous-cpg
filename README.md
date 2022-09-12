NEURON + Python model of the *Apteronotus leptorhynchus* pacemaker nucleus. Supporting code for:
  > Ilieş I, Zupanc GKH. Computational modeling predicts regulation of central pattern generator oscillations by size and density of the underlying heterogenous network. J Comput Neurosci, in press.

#### Overview
The pacemaker nucleus (Pn) of the weakly electric fish *Apteronotus leptorynchus* is a neural network comprised of 60-110 interconnected pacemaker cells and 15-30 relay cells conveying its output to electromotoneurons in the spinal cord. Morphological studies have found that axons of pacemaker cells synapse through gap junctions with a small number of relay cells and other pacemaker cells, while relay cells do not make synaptic contact with either pacemaker cells or other relay cells. 

Accordingly, the Pn neural network is modeled as a random directed graph with configurable size (counts of cells of each type) and density parameters (average numbers of connections of each type). Each graph node is a cell consisting of two cylindrical sections (somatic and axonal) with sizes derived from morphological data. Each section is divided into 1-μm long segments expressing Hodgkin-Huxley ion channels, with electrical properties based on previous modeling studies. Graph edges represent synapses, modeled here as half-gap objects connecting the last axonal segment of the presynaptic cell to the first somatic segment of the postsynaptic cell. 

#### Requirements
- NEURON 8.0 simulation environment with:
  - MPI support
  - HalfGap mod
- Python 3.8 and the following libraries:
  - NetworkX
  - NumPy
  - SciPy
  - Pandas
  - Matplotlib

#### Organization
- pn_model.py - main file, encompassing:
  - construction of Pn NEURON objects
  - construction of random Pn networks
  - calculation of frequency metrics
  - parameter grid setup & evaluation
- run_batch.py - perform current study
- vm_checks.py - get sample Vm traces
- all other files = HalfGap mod
