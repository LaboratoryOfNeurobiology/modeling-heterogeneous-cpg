/* Created by Language version: 7.7.0 */
/* NOT VECTORIZED */
#define NRN_VECTORIZED 0
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "scoplib_ansi.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__HalfGap
#define _nrn_initial _nrn_initial__HalfGap
#define nrn_cur _nrn_cur__HalfGap
#define _nrn_current _nrn_current__HalfGap
#define nrn_jacob _nrn_jacob__HalfGap
#define nrn_state _nrn_state__HalfGap
#define _net_receive _net_receive__HalfGap 
 
#define _threadargscomma_ /**/
#define _threadargsprotocomma_ /**/
#define _threadargs_ /**/
#define _threadargsproto_ /**/
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 static double *_p; static Datum *_ppvar;
 
#define t nrn_threads->_t
#define dt nrn_threads->_dt
#define nu _p[0]
#define isanode _p[1]
#define gmin _p[2]
#define gmax _p[3]
#define g _p[4]
#define i _p[5]
#define _g _p[6]
#define _nd_area  *_ppvar[0]._pval
#define vgap	*_ppvar[2]._pval
#define _p_vgap	_ppvar[2]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  2;
 /* external NEURON variables */
 /* declaration of user functions */
 static double _hoc_calcg();
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern Prop* nrn_point_prop_;
 static int _pointtype;
 static void* _hoc_create_pnt(_ho) Object* _ho; { void* create_point_process();
 return create_point_process(_pointtype, _ho);
}
 static void _hoc_destroy_pnt();
 static double _hoc_loc_pnt(_vptr) void* _vptr; {double loc_point_process();
 return loc_point_process(_pointtype, _vptr);
}
 static double _hoc_has_loc(_vptr) void* _vptr; {double has_loc_point();
 return has_loc_point(_vptr);
}
 static double _hoc_get_loc_pnt(_vptr)void* _vptr; {
 double get_loc_point_process(); return (get_loc_point_process(_vptr));
}
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 static void _hoc_setdata(void* _vptr) { Prop* _prop;
 _prop = ((Point_process*)_vptr)->_prop;
   _setdata(_prop);
 }
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 0,0
};
 static Member_func _member_func[] = {
 "loc", _hoc_loc_pnt,
 "has_loc", _hoc_has_loc,
 "get_loc", _hoc_get_loc_pnt,
 "calcg", _hoc_calcg,
 0, 0
};
#define calcg calcg_HalfGap
 extern double calcg( double );
 /* declare global and static user variables */
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "nu", "mV",
 "gmin", "nS",
 "gmax", "nS",
 "g", "nS",
 "i", "nA",
 "vgap", "mV",
 0,0
};
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(_NrnThread*, _Memb_list*, int);
static void nrn_state(_NrnThread*, _Memb_list*, int);
 static void nrn_cur(_NrnThread*, _Memb_list*, int);
static void  nrn_jacob(_NrnThread*, _Memb_list*, int);
 static void _hoc_destroy_pnt(_vptr) void* _vptr; {
   destroy_point_process(_vptr);
}
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"HalfGap",
 "nu",
 "isanode",
 "gmin",
 "gmax",
 0,
 "g",
 "i",
 0,
 0,
 "vgap",
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 7, _prop);
 	/*initialize range parameters*/
 	nu = 8;
 	isanode = 0;
 	gmin = 0.5;
 	gmax = 10;
  }
 	_prop->param = _p;
 	_prop->param_size = 7;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 3, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, _NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _HalfGap_reg() {
	int _vectorized = 0;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 0,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 7, 3);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "pointer");
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 HalfGap C:/Users/I/pn/HalfGap.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 
double calcg (  double _lvx ) {
   double _lcalcg;
 if ( ( isanode  != 1.0 )  && ( isanode  != - 1.0 ) ) {
     isanode = 0.0 ;
     printf ( "parameter isanode must be 1 for anode side of gap, -1 for cathode side\n" ) ;
     _lcalcg = 0.0 ;
     }
   else {
     _lcalcg = gmin + ( gmax - gmin ) / ( 1.0 + exp ( - isanode * _lvx / nu ) ) ;
     }
   
return _lcalcg;
 }
 
static double _hoc_calcg(void* _vptr) {
 double _r;
    _hoc_setdata(_vptr);
 _r =  calcg (  *getarg(1) );
 return(_r);
}

static void initmodel() {
  int _i; double _save;_ninits++;
{
 {
   g = calcg ( _threadargscomma_ v - vgap ) ;
   i = ( v - vgap ) * g * ( 0.001 ) ;
   }

}
}

static void nrn_init(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
 initmodel();
}}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   g = calcg ( _threadargscomma_ v - vgap ) ;
   i = ( v - vgap ) * g * ( 0.001 ) ;
   }
 _current += i;

} return _current;
}

static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 _g = _nrn_current(_v + .001);
 	{ _rhs = _nrn_current(_v);
 	}
 _g = (_g - _rhs)/.001;
 _g *=  1.e2/(_nd_area);
 _rhs *= 1.e2/(_nd_area);
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}}

static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}}

static void nrn_state(_NrnThread* _nt, _Memb_list* _ml, int _type){

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "HalfGap.mod";
static const char* nmodl_file_text = 
  "COMMENT\n"
  "Implements the rectifying gap junction described in\n"
  "\n"
  "@Article{Gutierrez2013,\n"
  "  author        = {Gutierrez, Gabrielle J. and Marder, Eve},\n"
  "  title         = {Rectifying electrical synapses can affect the influence of synaptic modulation on output pattern robustness},\n"
  "  journal       = {Journal of Neuroscience},\n"
  "  year          = {2013},\n"
  "  volume        = {33},\n"
  "  number        = {32},\n"
  "  pages         = {13238--13248},\n"
  "  doi           = {10.1523/JNEUROSCI.0937-13.2013},\n"
  "  eprint        = {http://www.jneurosci.org/content/33/32/13238.full.pdf},\n"
  "  pmid          = {23926276},\n"
  "}\n"
  "\n"
  "They assume a mechanism with instantaneous rectification\n"
  "in which the voltage dependence of gap conductance is\n"
  "described by an equation of the form\n"
  "\n"
  "g = gmin + (gmax - gmin)/( 1 + exp((v1 - v2)/nu) )\n"
  "\n"
  "In their formulation, if nu < 0, then\n"
  "g approaches gmax from below as v1 - v2 becomes increasingly positive,\n"
  "and\n"
  "it approaches gmin from above as v1 - v2 becomes increasingly negative.\n"
  "\n"
  "The electrical equivalent of the gap junction is this diode\n"
  "\n"
  "v1     v2\n"
  "o-->|--o\n"
  "\n"
  "in which the arrow indicates the preferred direction\n"
  "of (classical) current flow, i.e. the anode is on the left\n"
  "and the cathode is on the right.\n"
  "\n"
  "--------------------\n"
  "NMODL implementation\n"
  "--------------------\n"
  "\n"
  "The electrical effect of a gap junction can be implemented\n"
  "with a pair of point processes that are attached to the segments\n"
  "that are to be coupled.  Each point process monitors the\n"
  "potential of the segment to which it is attached,\n"
  "and delivers the appropriate current to that segment.\n"
  "\n"
  "Design decisions and notes:\n"
  "\n"
  "1.  Since two of these point processes are required\n"
  "to implement a single gap junction,\n"
  "it makes sense to call the mechanism class HalfGap.\n"
  "\n"
  "2.  Each instance of HalfGap will calculate the value of g\n"
  "from the difference between the membrane potential\n"
  "in the segment to which it is attached\n"
  "and the membrane potential in the segment on the\n"
  "opposite side of the gap junction;\n"
  "the value of the latter will be accessed via a POINTER.\n"
  "\n"
  "3.  The authors' original formula for g requires nu < 0\n"
  "for the HalfGap on the anode side of the gap junction,\n"
  "and nu > 0 for the HalfGap on the cathode side.\n"
  "That seems guaranteed to confuse users, so the formula\n"
  "employed by this NMODL implementation is\n"
  "\n"
  "g = gmin + (gmax - gmin)/( 1 + exp(-isanode*(v1 - v2)/nu) )\n"
  "\n"
  "where\n"
  "nu is always >= 0\n"
  "and\n"
  "isanode is a parameter whose value is 1\n"
  "to specify that the HalfGap is on the\n"
  "anode side of the gap junction,\n"
  "and -1 to specify that it is on the cathode side.\n"
  "\n"
  "4.  This current passes through ion channels in the cell membrane\n"
  "so its polarity convention is the same as for other transmembrane\n"
  "currents:  i < 0 depolarizes, > 0 hyperpolarizes.\n"
  "ENDCOMMENT\n"
  "\n"
  "NEURON {\n"
  "  POINT_PROCESS HalfGap\n"
  "  RANGE nu, isanode\n"
  "  RANGE gmin, gmax\n"
  "  NONSPECIFIC_CURRENT i\n"
  "  RANGE g, i\n"
  "  POINTER vgap : membrane potential on the \"other side\" of the junction\n"
  "    : i.e. the side to which the point process is NOT attached\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "  (mV) = (millivolt)\n"
  "  (nS) = (nanosiemens)\n"
  "  (nA) = (nanoamp)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "  nu = 8 (mV) : the magnitude of nu is inversely proportional\n"
  "    : to the voltage dependence of the gap junction's conductance\n"
  "    : i.e. large |nu| means less rectification\n"
  "  isanode = 0 : 1 means the HalfGap is on the anode side of the gap\n"
  "            : -1 means it's on the cathode side\n"
  "            : any other value generates an error message\n"
  "  gmin = 0.5 (nS)\n"
  "  gmax = 10 (nS)\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "  v (mV) : membrane potential in the segment to which the HalfGap is attached\n"
  "  g (nS) : gap junction conductance\n"
  "  vgap (mV) : membrane potential on the \"other side\" of the gap junction\n"
  "  i (nA) : current delivered by the HalfGap to the segment to which it is attached\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "  g = calcg(v - vgap)\n"
  "  i = (v - vgap)*g*(0.001)\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "  g = calcg(v - vgap)\n"
  "  i = (v - vgap)*g*(0.001)\n"
  "}\n"
  "\n"
  "FUNCTION calcg(vx(mV)) (nS) {\n"
  "  if ((isanode != 1) && (isanode != -1)) {\n"
  "    isanode = 0\n"
  "    printf(\"parameter isanode must be 1 for anode side of gap, -1 for cathode side\\n\")\n"
  "    calcg = 0\n"
  "  } else {\n"
  "    calcg = gmin + (gmax - gmin)/( 1 + exp(-isanode*vx/nu) )\n"
  "  }\n"
  "}\n"
  ;
#endif
