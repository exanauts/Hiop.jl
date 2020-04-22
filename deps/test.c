#include <mpi.h>
#include "hiopinterface.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    cHiopProblem nlp;
    hiop_createProblem(&nlp);
    hiop_solveProblem(&nlp);
    hiop_destroyProblem(&nlp);
    MPI_Finalize();
}