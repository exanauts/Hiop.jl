#include "chiopInterface.hpp"
extern "C" {

using namespace hiop;

int hiop_createProblem(cHiopProblem *prob) {
  cppJuliaProblem * cppproblem = new cppJuliaProblem(prob);
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
  prob->obj_value = solver.getObjective();
  solver.getSolution(prob->solution);
  return 0;
}

int hiop_destroyProblem(cHiopProblem *prob) {
  delete prob->refcppHiop;
  return 0;
}
} // extern C