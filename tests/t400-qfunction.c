/// @file
/// Test creation, evaluation, and destruction for qfunction
/// \test Test creation, evaluation, and destruction for qfunction
#include <ceed.h>

#include "t400-qfunction.h"

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector in[16], out[16];
  CeedVector Qdata, W, U, V;
  CeedQFunction qf_setup, qf_mass;
  CeedInt Q = 8;
  const CeedScalar *vv;
  CeedScalar w[Q], u[Q], v[Q];


  CeedInit(argv[1], &ceed);

  printf("HERE 1\n");

  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "w", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup, "qdata", 1, CEED_EVAL_NONE);

  printf("HERE 2\n");

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "qdata", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", 1, CEED_EVAL_INTERP);

  printf("HERE 3\n");

  for (CeedInt i=0; i<Q; i++) {
    CeedScalar x = 2.*i/(Q-1) - 1;
    w[i] = 1 - x*x;
    u[i] = 2 + 3*x + 5*x*x;
    v[i] = w[i] * u[i];
  }

  printf("HERE 4\n");

  CeedVectorCreate(ceed, Q, &W);
  CeedVectorSetArray(W, CEED_MEM_HOST, CEED_USE_POINTER, w);
  CeedVectorCreate(ceed, Q, &U);
  CeedVectorSetArray(U, CEED_MEM_HOST, CEED_USE_POINTER, u);
  CeedVectorCreate(ceed, Q, &V);
  CeedVectorSetValue(V, 0);
  CeedVectorCreate(ceed, Q, &Qdata);
  CeedVectorSetValue(Qdata, 0);

  printf("HERE 5\n");

  {
    in[0] = W;
    out[0] = Qdata;
    CeedQFunctionApply(qf_setup, Q, in, out);
  }

  printf("HERE 6\n");
  {
    in[0] = W;
    in[1] = U;
    out[0] = V;
    CeedQFunctionApply(qf_mass, Q, in, out);
  }

  printf("HERE 7\n");

  CeedVectorGetArrayRead(V, CEED_MEM_HOST, &vv);

  printf("HERE 8\n");
  for (CeedInt i=0; i<Q; i++)
    if (v[i] != vv[i])
      // LCOV_EXCL_START
      printf("[%d] v %f != vv %f\n",i, v[i], vv[i]);
  // LCOV_EXCL_STOP

  printf("HERE 9\n");
  CeedVectorRestoreArrayRead(V, &vv);

  CeedVectorDestroy(&W);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedVectorDestroy(&Qdata);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedDestroy(&ceed);
  return 0;
}
