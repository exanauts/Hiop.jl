#include <mpi.h>
#include "hiopinterface.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    void *nlp;
    hiop_createProblem(&nlp, 100);
    hiop_solveProblem(nlp);
    hiop_destroyProblem(nlp);
    MPI_Finalize();
}