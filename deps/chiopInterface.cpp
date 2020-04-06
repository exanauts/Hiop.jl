#include "chiopInterface.hpp"

using namespace hiop;
extern "C" int hiop_createProblem(void **ret, int ns) {
  cJuliaProblem * problem = (cJuliaProblem*) malloc(sizeof(cJuliaProblem));
  long long int n=2*ns+ns;
  long long int m=ns+3; 
  cppJuliaProblem * cppproblem = new cppJuliaProblem(ns, n, m);
  hiopNlpMDS *nlp = new hiopNlpMDS(*cppproblem);
  nlp->options->SetStringValue("dualsUpdateType", "linear");
  nlp->options->SetStringValue("dualsInitialization", "zero");

  nlp->options->SetStringValue("Hessian", "analytical_exact");
  nlp->options->SetStringValue("KKTLinsys", "xdycyd");
  nlp->options->SetStringValue("compute_mode", "hybrid");

  nlp->options->SetIntegerValue("verbosity_level", 3);
  nlp->options->SetNumericValue("mu0", 1e-1);
  problem->cppproblem = nlp;
  *ret = problem;
  return 0;
} 

extern "C" int hiop_solveProblem(void *problem) {
  hiopSolveStatus status;
  cJuliaProblem * cproblem = static_cast<cJuliaProblem*>(problem);
  hiopAlgFilterIPMNewton solver(cproblem->cppproblem);
  status = solver.run();
  double obj_value = solver.getObjective();
  return 0;
}

extern "C" int hiop_destroyProblem(void *problem) {
  cJuliaProblem * cproblem = static_cast<cJuliaProblem*>(problem);
  delete cproblem->cppproblem;
  return 0;
}