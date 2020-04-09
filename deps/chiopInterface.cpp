#include "chiopInterface.hpp"
extern "C" {

int (*hiop_eval_f_cb)(int n, double* x, int new_x, double* obj, cHiopProblem* cprob);

using namespace hiop;

int hiop_createProblem(cHiopProblem *prob, int ns) {
  long long int n=2*ns+ns;
  long long int m=ns+3; 
  cppJuliaProblem * cppproblem = new cppJuliaProblem(ns, n, m, prob);
  hiopNlpMDS *nlp = new hiopNlpMDS(*cppproblem);
  nlp->options->SetStringValue("dualsUpdateType", "linear");
  nlp->options->SetStringValue("dualsInitialization", "zero");

  nlp->options->SetStringValue("Hessian", "analytical_exact");
  nlp->options->SetStringValue("KKTLinsys", "xdycyd");
  nlp->options->SetStringValue("compute_mode", "hybrid");

  nlp->options->SetIntegerValue("verbosity_level", 3);
  nlp->options->SetNumericValue("mu0", 1e-1);
  prob->refcppHiop = nlp;
  return 0;
} 

int hiop_solveProblem(cHiopProblem *prob) {
  hiopSolveStatus status;
  hiopAlgFilterIPMNewton solver(prob->refcppHiop);
  status = solver.run();
  double obj_value = solver.getObjective();
  return 0;
}

int hiop_destroyProblem(cHiopProblem *prob) {
  delete prob->refcppHiop;
  return 0;
}
}