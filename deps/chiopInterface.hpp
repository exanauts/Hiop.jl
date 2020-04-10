#ifndef CHIOP_INTERFACE_HPP
#define CHIOP_INTERFACE_HPP
#include "hiop_defs.hpp"
#include "hiopInterface.hpp"
// Only included for now
#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"
#include "hiopMatrix.hpp"
#include <iostream>

using namespace hiop;
extern "C" {
  typedef struct cHiopProblem {
    hiopNlpMDS * refcppHiop;
    void *jprob;
    int (*get_prob_sizes)(long long* n_, long long* m_, void* jprob); 
    int (*get_vars_info)(long long n, double *xlow_, double* xupp_, void* jprob);
    int (*get_cons_info)(long long m, double *clow_, double* cupp_, void* jprob);
    int (*eval_f)(int n, double* x, int new_x, double* obj, void* jprob);
    int (*eval_grad_f)(long long n, double* x, int new_x, double* gradf, void* jprob);
    int (*eval_cons)(long long n, long long m,
      const long long num_cons, long long* idx_cons,  
      double* x, int new_x, 
      double* cons, void* jprob);
    int (*get_sparse_dense_blocks_info)(int* nx_sparse, int* nx_dense,
      int* nnz_sparse_Jaceq, int* nnz_sparse_Jacineq,
      int* nnz_sparse_Hess_Lagr_SS, 
      int* nnz_sparse_Hess_Lagr_SD, void* jprob);
    int (*eval_Jac_cons)(long long n, long long m,
      long long num_cons, long long* idx_cons,
      double* x, int new_x,
      long long nsparse, long long ndense, 
      int nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
      double** JacD, void *jprob);
    int (*eval_Hess_Lagr)(long long n, long long m,
      double* x, int new_x, double obj_factor,
      double* lambda, int new_lambda,
      long long nsparse, long long ndense, 
      int nnzHSS, int* iHSS, int* jHSS, double* MHSS, 
      double** HDD,
      int nnzHSD, int* iHSD, int* jHSD, double* MHSD, void* jprob);
  } cHiopProblem;
}


class cppJuliaProblem : public hiopInterfaceMDS
{
  public:
    cppJuliaProblem(int ns_, long long int n_, long long int m_, cHiopProblem *cprob_)
      : ns(ns_), n(n_), m(m_), cprob(cprob_) 
    {
      if(ns<=0) {
        ns = 4;
      } else {
        if(4*(ns/4) != ns) {
          ns = 4*((4+ns)/4);
          printf("[warning] number (%d) of sparse vars is not a multiple ->was altered to %d\n", 
          ns_, ns); 
        }
      }
      nd = ns;

      Q  = new hiopMatrixDense(nd,nd);
      Q->setToConstant(1e-8);
      Q->addDiagonal(2.);
      double** Qa = Q->get_M();
      for(int i=1; i<nd-1; i++) {
        Qa[i][i+1] = 1.;
        Qa[i+1][i] = 1.;
      }

      Md = new hiopMatrixDense(ns,nd);
      Md->setToConstant(-1.0);

      _buf_y = new double[nd];
    }

    virtual ~cppJuliaProblem()
    {
      delete[] _buf_y;
      delete Md;
      delete Q;
    }

    bool get_prob_sizes(long long& n_, long long& m_) 
    {
      // n_ = n;
      // m_ = m;
      cprob->get_prob_sizes(&n_, &m_, cprob->jprob);
      return true;
    };
    bool get_vars_info(const long long& n, double *xlow_, double* xupp_, NonlinearityType* type)
    {
      // assert(n>=4 && "number of variables should be greater than 4 for this example");
      // assert(n==2*ns+nd);

      // //x
      // for(int i=0; i<ns; ++i) xlow_[i] = -1e+20;
      // //s
      // for(int i=ns; i<2*ns; ++i) xlow_[i] = 0.;
      // //y 
      // xlow_[2*ns] = -4.;
      // for(int i=2*ns+1; i<n; ++i) xlow_[i] = -1e+20;
      
      // //x
      // for(int i=0; i<ns; ++i) xupp_[i] = 3.;
      // //s
      // for(int i=ns; i<2*ns; ++i) xupp_[i] = +1e+20;
      // //y
      // xupp_[2*ns] = 4.;
      // for(int i=2*ns+1; i<n; ++i) xupp_[i] = +1e+20;

      for(int i=0; i<n; ++i) type[i]=hiopNonlinear;
      cprob->get_vars_info(n, xlow_, xupp_, cprob->jprob);
      return true;
    };
    bool get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type)
    {
      // assert(m==ns+3);
      // int i;
      // //x+s - Md y = 0, i=1,...,ns
      // for(i=0; i<ns; i++) clow[i] = cupp[i] = 0.;

      // // [-2  ]    [ x_1 + e^T s]   [e^T]      [ 2 ]
      // clow[i] = -2; cupp[i++] = 2.;
      // // [-inf] <= [ x_2        ] + [e^T] y <= [ 2 ]
      // clow[i] = -1e+20; cupp[i++] = 2.;
      // // [-2  ]    [ x_3        ]   [e^T]      [inf]
      // clow[i] = -2; cupp[i++] = 1e+20;
      // assert(i==m);

      for(int i=0; i<m; ++i) type[i]=hiopNonlinear;
      cprob->get_cons_info(m, clow, cupp, cprob->jprob);
      return true;
    };
    bool eval_f(const long long& n, const double* x, bool new_x, double& obj_value)
    {
      // assert(ns>=4); assert(Q->n()==nd); assert(Q->m()==nd);
      // obj_value=x[0]*(x[0]-1.);
      // //sum 0.5 {x_i*(x_{i}-1) : i=1,...,ns} + 0.5 y'*Qd*y + 0.5 s^T s
      // for(int i=1; i<ns; i++) obj_value += x[i]*(x[i]-1.);
      // obj_value *= 0.5;
      // std::cout.precision(17);  
      // std::cout << "term1: " << obj_value << std::endl;

      // double term2=0.;
      // const double* y = x+2*ns;
      // Q->timesVec(0.0, _buf_y, 1., y);
      // for(int i=0; i<nd; i++) term2 += _buf_y[i] * y[i];
      // obj_value += 0.5*term2;
      // std::cout << "term2: " << obj_value << std::endl;
      
      // const double* s=x+ns;
      // double term3=s[0]*s[0];
      // for(int i=1; i<ns; i++) term3 += s[i]*s[i];
      // obj_value += 0.5*term3;
      // std::cout << "term3: " << obj_value << std::endl;
      cprob->eval_f(n, (double *) x, 0, &obj_value, cprob->jprob);
      return true;
    };

    //sum 0.5 {x_i*(x_{i}-1) : i=1,...,ns} + 0.5 y'*Qd*y + 0.5 s^T s
    bool eval_grad_f(const long long& n, const double* x, bool new_x, double* gradf)
    {
      //! assert(ns>=4); assert(Q->n()==ns/4); assert(Q->m()==ns/4);
      //x_i - 0.5 
      for(int i=0; i<ns; i++) 
        gradf[i] = x[i]-0.5;

      //Qd*y
      const double* y = x+2*ns;
      double* gradf_y = gradf+2*ns;
      Q->timesVec(0.0, gradf_y, 1., y);

      //s
      const double* s=x+ns;
      double* gradf_s = gradf+ns;
      for(int i=0; i<ns; i++) gradf_s[i] = s[i];
      cprob->eval_grad_f(n, (double *) x, 0, gradf, cprob->jprob);

      return true;
    };
    bool eval_cons(const long long& n, const long long& m,
      const long long& num_cons, const long long* idx_cons,  
      const double* x, bool new_x, 
      double* cons)
    {
      cprob->eval_cons(n, m, num_cons, (long long *) idx_cons, (double *) x, new_x, cons, cprob->jprob);
      return true;
      const double* s = x+ns;
      const double* y = x+2*ns;

      assert(num_cons==ns || num_cons==3);

      bool isEq=false;
      for(int irow=0; irow<num_cons; irow++) {
        const int con_idx = (int) idx_cons[irow];
        if(con_idx<ns) {
          //equalities: x+s - Md y = 0
          cons[con_idx] = x[con_idx] + s[con_idx];
          isEq=true;
        } else {
          assert(con_idx<ns+3);
          //inequality
          const int conineq_idx=con_idx-ns;
          if(conineq_idx==0) {
            cons[conineq_idx] = x[0];
            for(int i=0; i<ns; i++)   cons[conineq_idx] += s[i];
            for(int i=0; i<nd; i++) cons[conineq_idx] += y[i];
          } else if(conineq_idx==1) {
            cons[conineq_idx] = x[1];
            for(int i=0; i<nd; i++) cons[conineq_idx] += y[i];
          } else if(conineq_idx==2) {
            cons[conineq_idx] = x[2];
            for(int i=0; i<nd; i++) cons[conineq_idx] += y[i];
          } else { assert(false); }
        }  
      }
      if(isEq) {
        Md->timesVec(1.0, cons, 1.0, y);
      }
      return true;
    };
    bool get_sparse_dense_blocks_info(int& nx_sparse, int& nx_dense,
      int& nnz_sparse_Jaceq, int& nnz_sparse_Jacineq,
      int& nnz_sparse_Hess_Lagr_SS, 
      int& nnz_sparse_Hess_Lagr_SD)
    {
      // nx_sparse = 2*ns;
      // nx_dense = nd;
      // nnz_sparse_Jaceq = 2*ns;
      // nnz_sparse_Jacineq = 1+ns+1+1;
      // nnz_sparse_Hess_Lagr_SS = 2*ns;
      // nnz_sparse_Hess_Lagr_SD = 0.;
      cprob->get_sparse_dense_blocks_info(&nx_sparse, &nx_dense, &nnz_sparse_Jaceq, &nnz_sparse_Jacineq, &nnz_sparse_Hess_Lagr_SS, &nnz_sparse_Hess_Lagr_SD, cprob->jprob);
      return true;
    };
    bool eval_Jac_cons(const long long& n, const long long& m,
      const long long& num_cons, const long long* idx_cons,
      const double* x, bool new_x,
      const long long& nsparse, const long long& ndense, 
      const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
      double** JacD)
    {
      //const double* s = x+ns;
      //const double* y = x+2*ns;

      assert(num_cons==ns || num_cons==3);

      if(iJacS!=NULL && jJacS!=NULL) {
        int nnzit=0;
        for(int itrow=0; itrow<num_cons; itrow++) {
          const int con_idx = (int) idx_cons[itrow];
          if(con_idx<ns) {
            //sparse Jacobian eq w.r.t. x and s
            //x
            iJacS[nnzit] = con_idx;
            jJacS[nnzit] = con_idx;
            nnzit++;

            //s
            iJacS[nnzit] = con_idx;
            jJacS[nnzit] = con_idx+ns;
            nnzit++;

          } else {
            //sparse Jacobian ineq w.r.t x and s
            if(con_idx-ns==0) {
              //w.r.t x_1
              iJacS[nnzit] = 0;
              jJacS[nnzit] = 0;
              nnzit++;
              //w.r.t s
              for(int i=0; i<ns; i++) {
                iJacS[nnzit] = 0;
                jJacS[nnzit] = ns+i;
                nnzit++;
              }
            } else {
              assert(con_idx-ns==1 || con_idx-ns==2);
              //w.r.t x_2 or x_3
              iJacS[nnzit] = con_idx-ns;
              jJacS[nnzit] = con_idx-ns;
              nnzit++;
            }
          }
        }
        assert(nnzit==nnzJacS);
      } 
      //values for sparse Jacobian if requested by the solver
      if(MJacS!=NULL) {
      int nnzit=0;
      for(int itrow=0; itrow<num_cons; itrow++) {
        const int con_idx = (int) idx_cons[itrow];
        if(con_idx<ns) {
          //sparse Jacobian EQ w.r.t. x and s
          //x
          MJacS[nnzit] = 1.;
          nnzit++;
          
          //s
          MJacS[nnzit] = 1.;
          nnzit++;
          
        } else {
          //sparse Jacobian INEQ w.r.t x and s
          if(con_idx-ns==0) {
            //w.r.t x_1
            MJacS[nnzit] = 1.;
            nnzit++;
            //w.r.t s
            for(int i=0; i<ns; i++) {
              MJacS[nnzit] = 1.;
              nnzit++;
            }
          } else {
            assert(con_idx-ns==1 || con_idx-ns==2);
            //w.r.t x_2 or x_3
            MJacS[nnzit] = 1.;
            nnzit++;
          }
        }
      }
      assert(nnzit==nnzJacS);
      }
      
      //dense Jacobian w.r.t y
      if(JacD!=NULL) {
        bool isEq=false;
        for(int itrow=0; itrow<num_cons; itrow++) {
          const int con_idx = (int) idx_cons[itrow];
          if(con_idx<ns) {
            isEq=true;
            assert(num_cons==ns);
            continue;
          } else {
            //do an in place fill-in for the ineq Jacobian corresponding to e^T
            assert(con_idx-ns==0 || con_idx-ns==1 || con_idx-ns==2);
            assert(num_cons==3);
            for(int i=0; i<nd; i++) {
              JacD[con_idx-ns][i] = 1.;
            }
          }
        }
        if(isEq) {
          memcpy(JacD[0], Md->local_buffer(), ns*nd*sizeof(double));
        }
      }
      return true;
      };
    bool eval_Hess_Lagr(const long long& n, const long long& m,
      const double* x, bool new_x, const double& obj_factor,
      const double* lambda, bool new_lambda,
      const long long& nsparse, const long long& ndense, 
      const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS, 
      double** HDD,
      int& nnzHSD, int* iHSD, int* jHSD, double* MHSD)
    {
      //Note: lambda is not used since all the constraints are linear and, therefore, do 
      //not contribute to the Hessian of the Lagrangian

      assert(nnzHSS==2*ns);
      assert(nnzHSD==0);
      assert(iHSD==NULL); assert(jHSD==NULL); assert(MHSD==NULL);

      if(iHSS!=NULL && jHSS!=NULL) {
        for(int i=0; i<2*ns; i++) iHSS[i] = jHSS[i] = i;     
      }

      if(MHSS!=NULL) {
        for(int i=0; i<2*ns; i++) MHSS[i] = obj_factor;
      }

      if(HDD!=NULL) {
        const int nx_dense_squared = nd*nd;
        //memcpy(HDD[0], Q->local_buffer(), nx_dense_squared*sizeof(double));
        const double* Qv = Q->local_buffer();
        for(int i=0; i<nx_dense_squared; i++)
          HDD[0][i] = obj_factor*Qv[i];
      }
      return true;
    };
private:
  int ns, nd;
  long long int n, m;
  hiop::hiopMatrixDense *Q, *Md;
  double* _buf_y;
  cHiopProblem *cprob;
};

extern "C" int hiop_createProblem(cHiopProblem *problem, int ns);
extern "C" int hiop_solveProblem(cHiopProblem *problem);
extern "C" int hiop_destroyProblem(cHiopProblem *problem);
#endif