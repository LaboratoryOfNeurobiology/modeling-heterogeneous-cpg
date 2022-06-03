COMMENT
Implements the rectifying gap junction described in

@Article{Gutierrez2013,
  author        = {Gutierrez, Gabrielle J. and Marder, Eve},
  title         = {Rectifying electrical synapses can affect the influence of synaptic modulation on output pattern robustness},
  journal       = {Journal of Neuroscience},
  year          = {2013},
  volume        = {33},
  number        = {32},
  pages         = {13238--13248},
  doi           = {10.1523/JNEUROSCI.0937-13.2013},
  eprint        = {http://www.jneurosci.org/content/33/32/13238.full.pdf},
  pmid          = {23926276},
}

They assume a mechanism with instantaneous rectification
in which the voltage dependence of gap conductance is
described by an equation of the form

g = gmin + (gmax - gmin)/( 1 + exp((v1 - v2)/nu) )

In their formulation, if nu < 0, then
g approaches gmax from below as v1 - v2 becomes increasingly positive,
and
it approaches gmin from above as v1 - v2 becomes increasingly negative.

The electrical equivalent of the gap junction is this diode

v1     v2
o-->|--o

in which the arrow indicates the preferred direction
of (classical) current flow, i.e. the anode is on the left
and the cathode is on the right.

--------------------
NMODL implementation
--------------------

The electrical effect of a gap junction can be implemented
with a pair of point processes that are attached to the segments
that are to be coupled.  Each point process monitors the
potential of the segment to which it is attached,
and delivers the appropriate current to that segment.

Design decisions and notes:

1.  Since two of these point processes are required
to implement a single gap junction,
it makes sense to call the mechanism class HalfGap.

2.  Each instance of HalfGap will calculate the value of g
from the difference between the membrane potential
in the segment to which it is attached
and the membrane potential in the segment on the
opposite side of the gap junction;
the value of the latter will be accessed via a POINTER.

3.  The authors' original formula for g requires nu < 0
for the HalfGap on the anode side of the gap junction,
and nu > 0 for the HalfGap on the cathode side.
That seems guaranteed to confuse users, so the formula
employed by this NMODL implementation is

g = gmin + (gmax - gmin)/( 1 + exp(-isanode*(v1 - v2)/nu) )

where
nu is always >= 0
and
isanode is a parameter whose value is 1
to specify that the HalfGap is on the
anode side of the gap junction,
and -1 to specify that it is on the cathode side.

4.  This current passes through ion channels in the cell membrane
so its polarity convention is the same as for other transmembrane
currents:  i < 0 depolarizes, > 0 hyperpolarizes.
ENDCOMMENT

NEURON {
  POINT_PROCESS HalfGap
  RANGE nu, isanode
  RANGE gmin, gmax
  NONSPECIFIC_CURRENT i
  RANGE g, i
  POINTER vgap : membrane potential on the "other side" of the junction
    : i.e. the side to which the point process is NOT attached
}

UNITS {
  (mV) = (millivolt)
  (nS) = (nanosiemens)
  (nA) = (nanoamp)
}

PARAMETER {
  nu = 8 (mV) : the magnitude of nu is inversely proportional
    : to the voltage dependence of the gap junction's conductance
    : i.e. large |nu| means less rectification
  isanode = 0 : 1 means the HalfGap is on the anode side of the gap
            : -1 means it's on the cathode side
            : any other value generates an error message
  gmin = 0.5 (nS)
  gmax = 10 (nS)
}

ASSIGNED {
  v (mV) : membrane potential in the segment to which the HalfGap is attached
  g (nS) : gap junction conductance
  vgap (mV) : membrane potential on the "other side" of the gap junction
  i (nA) : current delivered by the HalfGap to the segment to which it is attached
}

INITIAL {
  g = calcg(v - vgap)
  i = (v - vgap)*g*(0.001)
}

BREAKPOINT {
  g = calcg(v - vgap)
  i = (v - vgap)*g*(0.001)
}

FUNCTION calcg(vx(mV)) (nS) {
  if ((isanode != 1) && (isanode != -1)) {
    isanode = 0
    printf("parameter isanode must be 1 for anode side of gap, -1 for cathode side\n")
    calcg = 0
  } else {
    calcg = gmin + (gmax - gmin)/( 1 + exp(-isanode*vx/nu) )
  }
}
