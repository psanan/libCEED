//                        libCEED + PETSc Example: BP3
//
// This example demonstrates a simple usage of libCEED with PETSc to solve the
// CEED BP3 benchmark problem, see http://ceed.exascaleproject.org/bps.
//
// The code is intentionally "raw", using only low-level communication
// primitives.
//
// Build with:
//
//     make bp3 [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//
//     bp3
//     bp3 -ceed /cpu/self
//     bp3 -ceed /gpu/occa
//     bp3 -ceed /cpu/occa
//     bp3 -ceed /omp/occa
//     bp3 -ceed /ocl/occa
//
//TESTARGS -ceed {ceed_resource} -test -degree 3

/// @file
/// Diffusion operator example using PETSc
const char help[] = "Solve CEED BP3 using PETSc\n";

#include <stdbool.h>
#include <petscdmplex.h>
#include "bp3.h"

typedef struct User_ *User;
struct User_ {
  MPI_Comm comm;
  Vec Xloc, Yloc;
  CeedVector xceed, yceed;
  CeedOperator op;
  CeedVector rho;
  Ceed ceed;
  DM   dm; 
};


static int CreateRestrictionPlex(Ceed ceed, CeedInt P, CeedInt ncomp,
				 CeedElemRestriction *Erestrict, DM dm) {

  PetscSection   section;
  PetscInt       c, cStart, cEnd, Nelem, Ndof, *erestrict, eoffset;
  PetscInt       ierr;
  Vec Uloc;

  PetscFunctionBegin;
  ierr = DMGetDefaultSection(dm,&section);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);

  Nelem = cEnd - cStart;
  ierr = PetscMalloc1(Nelem*P*P*P, &erestrict);CHKERRQ(ierr);
  for (c=cStart,eoffset=0; c<cEnd; c++) {
    PetscInt numindices,*indices,i;
    ierr = DMPlexGetClosureIndices(dm,section,section,c,&numindices,&indices,NULL);CHKERRQ(ierr);
    for (i=0; i<numindices; i+=ncomp) {
      for (PetscInt j=0; j<ncomp; j++) {
        if (indices[i+j] != indices[i]+j) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Cell %D closure indices not interlaced",c);
      }
      erestrict[eoffset++] = indices[i]/ncomp;
    }
    ierr = DMPlexRestoreClosureIndices(dm,section,section,c,&numindices,&indices,NULL);CHKERRQ(ierr);
  }
  ierr = DMGetLocalVector(dm, &Uloc);CHKERRQ(ierr);
  ierr = VecGetLocalSize(Uloc, &Ndof);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &Uloc);CHKERRQ(ierr);
  CeedElemRestrictionCreate(ceed, Nelem, P*P*P, Ndof/ncomp, ncomp,
                            CEED_MEM_HOST, CEED_COPY_VALUES, erestrict, Erestrict);
  ierr = PetscFree(erestrict);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
// This function uses libCEED to compute the action of the Laplacian with
// Dirichlet boundary conditions
static PetscErrorCode MatMult_Diff(Mat A, Vec X, Vec Y) {
  PetscErrorCode ierr;
  User user;
  PetscScalar *x, *y;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);
 
  DMGlobalToLocalBegin(user->dm, X, INSERT_VALUES, user->Xloc);
  DMGlobalToLocalEnd(user->dm, X, INSERT_VALUES, user->Xloc);

  DMGlobalToLocalBegin(user->dm, Y, INSERT_VALUES, user->Yloc);
  DMGlobalToLocalEnd(user->dm, Y, INSERT_VALUES, user->Yloc);

  ierr = VecZeroEntries(user->Yloc); CHKERRQ(ierr);

  ierr = VecGetArrayRead(user->Xloc, (const PetscScalar **)&x); CHKERRQ(ierr);
  ierr = VecGetArray(user->Yloc, &y); CHKERRQ(ierr);
  CeedVectorSetArray(user->xceed, CEED_MEM_HOST, CEED_USE_POINTER, x);
  CeedVectorSetArray(user->yceed, CEED_MEM_HOST, CEED_USE_POINTER, y);

  CeedOperatorApply(user->op, user->xceed, user->yceed,
                    CEED_REQUEST_IMMEDIATE);
  ierr = CeedVectorSyncArray(user->yceed, CEED_MEM_HOST); CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(user->Xloc, (const PetscScalar **)&x); CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Yloc, &y); CHKERRQ(ierr);

  ierr = VecZeroEntries(Y); CHKERRQ(ierr);
  DMLocalToGlobalBegin(user->dm, user->Yloc, INSERT_VALUES, Y);
  DMLocalToGlobalEnd(user->dm, user->Yloc, INSERT_VALUES, Y);
 
  PetscFunctionReturn(0);
}


static PetscErrorCode ComputeErrorMax(User user, CeedOperator op_error, Vec X,
                                      CeedVector target, PetscReal *maxerror) {
  PetscErrorCode ierr;
  PetscScalar *x;
  CeedVector collocated_error;
  CeedInt length;

  PetscFunctionBeginUser;
  CeedVectorGetLength(target, &length);
  CeedVectorCreate(user->ceed, length, &collocated_error);
  ierr = DMGlobalToLocal(user->dm, X, INSERT_VALUES, user->Xloc);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->Xloc, (const PetscScalar **)&x); CHKERRQ(ierr);
  CeedVectorSetArray(user->xceed, CEED_MEM_HOST, CEED_USE_POINTER, x);
  CeedOperatorApply(op_error, user->xceed, collocated_error,
                    CEED_REQUEST_IMMEDIATE);
  VecRestoreArrayRead(user->Xloc, (const PetscScalar **)&x); CHKERRQ(ierr);

  *maxerror = 0;
  const CeedScalar *e;
  CeedVectorGetArrayRead(collocated_error, CEED_MEM_HOST, &e);
  for (CeedInt i=0; i<length; i++) {
    *maxerror = PetscMax(*maxerror, PetscAbsScalar(e[i]));
  }
  CeedVectorRestoreArrayRead(collocated_error, &e);
  ierr = MPI_Allreduce(MPI_IN_PLACE, &maxerror,
                       1, MPIU_SCALAR, MPIU_MAX, user->comm); CHKERRQ(ierr);
  CeedVectorDestroy(&collocated_error);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  PetscInt ierr;
  MPI_Comm comm;
  char ceedresource[4096] = "/cpu/self/ref/serial";
  PetscInt degree, qextra, melem[3] = {2, 1, 1}, lsize, gsize, dim;
  PetscScalar *r;
  PetscBool test_mode, benchmark_mode;
  Ceed ceed;
  CeedBasis basisx, basisu;
  CeedElemRestriction Erestrictx, Erestrictu, Erestrictxi, Erestrictui,
                      Erestrictqdi;
  CeedQFunction qf_setup, qf_diff, qf_error;
  CeedOperator op_setup, op_diff, op_error;
  CeedVector xcoord, rho, rhsceed, target;
  CeedInt P, Q;
  Vec X, Xloc, rhs, rhsloc;
  Mat mat;
  KSP ksp;
  DM  dm, dmcoord; 
  User user;
  double my_rt_start, my_rt, rt_min, rt_max;
  PetscInt		cStart, cEnd, nelem, marker_ids[] = {1};
  const PetscScalar     *coordArray;
  Vec			coords;
  PetscSpace            sp;
  PetscFE               fe;
  PetscSection          section;

  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, NULL, "CEED BP3 in PETSc", NULL); CHKERRQ(ierr);
  test_mode = PETSC_FALSE;
  ierr = PetscOptionsBool("-test",
                          "Testing mode (do not print unless error is large)",
                          NULL, test_mode, &test_mode, NULL); CHKERRQ(ierr);
  benchmark_mode = PETSC_FALSE;
  ierr = PetscOptionsBool("-benchmark",
                          "Benchmarking mode (prints benchmark statistics)",
                          NULL, benchmark_mode, &benchmark_mode, NULL);
  CHKERRQ(ierr);
  degree = 2; //test_mode ? 3 : 1;
  //ierr = PetscOptionsInt("-degree", "Polynomial degree of tensor product basis",
  //                       NULL, degree, &degree, NULL); CHKERRQ(ierr);
  qextra = 0;
  ierr = PetscOptionsInt("-qextra", "Number of extra quadrature points",
                         NULL, qextra, &qextra, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-ceed", "CEED resource specifier",
                            NULL, ceedresource, ceedresource,
                            sizeof(ceedresource), NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  CeedInit(ceedresource, &ceed);

  ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, "3dhole1z.exo", PETSC_TRUE, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);
  ierr = PetscFECreateDefault(PETSC_COMM_SELF,dim,1,PETSC_FALSE,NULL,PETSC_DETERMINE,&fe);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMAddField(dm,NULL,(PetscObject)fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
 //if (0) {
 //   ierr = DMAddBoundary(dm,DM_BC_ESSENTIAL,"wall","marker",0,0,NULL,(void(*)(void))TrueSolution,1,marker_ids,NULL);CHKERRQ(ierr);
 // }
  ierr = DMPlexSetClosurePermutationTensor(dm,PETSC_DETERMINE,NULL);CHKERRQ(ierr);

  ierr = PetscFEGetBasisSpace(fe, &sp);CHKERRQ(ierr);
  ierr = PetscSpaceGetDegree(sp, &degree, NULL);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  //if (degree < 1) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"-petscspace_degree %D must be at least 1",degree);

  P = degree + 1;
  Q = P + qextra;
  
  CeedBasisCreateTensorH1Lagrange(ceed, 3, 1, P, Q, CEED_GAUSS, &basisu);
  CeedBasisCreateTensorH1Lagrange(ceed, 3, 3, 2, Q, CEED_GAUSS, &basisx);

  ierr = DMGetCoordinateDM(dm, &dmcoord);CHKERRQ(ierr);
  ierr = DMPlexSetClosurePermutationTensor(dmcoord,PETSC_DETERMINE,NULL);CHKERRQ(ierr);

  CreateRestrictionPlex(ceed, 2, 3, &Erestrictx, dmcoord);
  CreateRestrictionPlex(ceed, P, 1, &Erestrictu, dm);

  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  nelem = cEnd - cStart;

  CeedElemRestrictionCreateIdentity(ceed, nelem, Q*Q*Q, nelem*Q*Q*Q, 1,
                                    &Erestrictui);
  CeedElemRestrictionCreateIdentity(ceed, nelem, Q*Q*Q, nelem*Q*Q*Q, 1,
                                    &Erestrictxi);
  CeedElemRestrictionCreateIdentity(ceed, nelem, 6*Q*Q*Q, 6*nelem*Q*Q*Q, 1,
                                    &Erestrictqdi);

  ierr = DMGetCoordinatesLocal(dm, &coords);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coords,&coordArray);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dmcoord, &section);CHKERRQ(ierr); 

  CeedElemRestrictionCreateVector(Erestrictx, &xcoord, NULL);
  CeedVectorSetArray(xcoord, CEED_MEM_HOST, CEED_COPY_VALUES, (PetscScalar*)coordArray);
  ierr = VecRestoreArrayRead(coords,&coordArray);CHKERRQ(ierr);

  DMCreateGlobalVector(dm,&X);
  DMCreateLocalVector(dm,&Xloc);
 
  PetscInt Xlocsize;
  VecGetSize(Xloc,&Xlocsize);
  VecGetLocalSize(X,&lsize);
  VecGetSize(X, &gsize); 
 
  // Create the Q-function that builds the diff operator (i.e. computes its
  // quadrature data) and set its context data.
  CeedQFunctionCreateInterior(ceed, 1,
                              Setup, __FILE__ ":Setup", &qf_setup);
  CeedQFunctionAddInput(qf_setup, "x", 3, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_setup, "dx", 3, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup, "rho", 6, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_setup, "true_soln", 1, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_setup, "rhs", 1, CEED_EVAL_INTERP);

  // Create the Q-function that defines the action of the diff operator.
  CeedQFunctionCreateInterior(ceed, 1,
                              Diff, __FILE__ ":Diff", &qf_diff);
  CeedQFunctionAddInput(qf_diff, "u", 1, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_diff, "rho", 6, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_diff, "v", 1, CEED_EVAL_GRAD);

  // Create the error qfunction
  CeedQFunctionCreateInterior(ceed, 1,
                              Error, __FILE__ ":Error", &qf_error);
  CeedQFunctionAddInput(qf_error, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_error, "true_soln", 1, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_error, "error", 1, CEED_EVAL_NONE);

  // Create the persistent vectors that will be needed in setup
  CeedInt Nqpts;
  CeedBasisGetNumQuadraturePoints(basisu, &Nqpts);
  CeedVectorCreate(ceed, 6*nelem*Nqpts, &rho);
  CeedVectorCreate(ceed, nelem*Nqpts, &target);
  CeedVectorCreate(ceed, lsize, &rhsceed);

  // Create the operator that builds the quadrature data for the diff operator.
  CeedOperatorCreate(ceed, qf_setup, NULL, NULL, &op_setup);
  CeedOperatorSetField(op_setup, "x", Erestrictx, CEED_NOTRANSPOSE,
                       basisx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "dx", Erestrictx, CEED_NOTRANSPOSE,
                       basisx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "weight", Erestrictxi, CEED_NOTRANSPOSE,
                       basisx, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "rho", Erestrictqdi, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "true_soln", Erestrictui, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, target);
  CeedOperatorSetField(op_setup, "rhs", Erestrictu, CEED_NOTRANSPOSE,
                       basisu, rhsceed);

  // Create the diff operator.
  CeedOperatorCreate(ceed, qf_diff, NULL, NULL, &op_diff);
  CeedOperatorSetField(op_diff, "u", Erestrictu, CEED_NOTRANSPOSE,
                       basisu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_diff, "rho", Erestrictqdi, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, rho);
  CeedOperatorSetField(op_diff, "v", Erestrictu, CEED_NOTRANSPOSE,
                       basisu, CEED_VECTOR_ACTIVE);

  // Create the error operator
  CeedOperatorCreate(ceed, qf_error, NULL, NULL, &op_error);
  CeedOperatorSetField(op_error, "u", Erestrictu, CEED_NOTRANSPOSE,
                       basisu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_error, "true_soln", Erestrictui, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, target);
  CeedOperatorSetField(op_error, "error", Erestrictui, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);


  // Set up Mat
  ierr = PetscMalloc1(1, &user); CHKERRQ(ierr);
  user->comm = comm;
  user->dm =dm; 
  user->Xloc = Xloc;
  ierr = VecDuplicate(Xloc, &user->Yloc); CHKERRQ(ierr);
  CeedVectorCreate(ceed, Xlocsize, &user->xceed);
  CeedVectorCreate(ceed, Xlocsize, &user->yceed);
  user->op = op_diff;
  user->rho = rho;
  user->ceed = ceed;

  ierr = MatCreateShell(comm, lsize, lsize, gsize, gsize, user, &mat); CHKERRQ(ierr);
  ierr = MatShellSetOperation(mat, MATOP_MULT, (void(*)(void))MatMult_Diff);

  ierr = VecDuplicate(X, &rhs); CHKERRQ(ierr);

  // Get RHS vector
  ierr = VecDuplicate(Xloc, &rhsloc); CHKERRQ(ierr);
  ierr = VecZeroEntries(rhsloc); CHKERRQ(ierr);
  ierr = VecGetArray(rhsloc, &r); CHKERRQ(ierr);

  CeedVectorSetArray(rhsceed, CEED_MEM_HOST, CEED_USE_POINTER, r);
  CeedOperatorApply(op_setup, xcoord, rho, CEED_REQUEST_IMMEDIATE);
  ierr = CeedVectorSyncArray(rhsceed, CEED_MEM_HOST); CHKERRQ(ierr);
  CeedVectorDestroy(&xcoord);
  // Gather RHS
  ierr = VecRestoreArray(rhsloc, &r); CHKERRQ(ierr);
  ierr = VecZeroEntries(rhs); CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(dm, rhsloc, ADD_VALUES, rhs);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, rhsloc, ADD_VALUES, rhs);CHKERRQ(ierr);

  CeedVectorDestroy(&rhsceed);
 
  ierr = KSPCreate(comm, &ksp); CHKERRQ(ierr);
  {
    PC pc;
    ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
    ierr = PCSetType(pc, PCNONE); CHKERRQ(ierr);
    ierr = KSPSetType(ksp, KSPCG); CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT,
                            PETSC_DEFAULT); CHKERRQ(ierr);
  }
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, mat, mat); CHKERRQ(ierr);
  // Timed solve
  my_rt_start = MPI_Wtime();
  ierr = KSPSolve(ksp, rhs, X); CHKERRQ(ierr);
  my_rt = MPI_Wtime() - my_rt_start;
  {
    KSPType ksptype;
    KSPConvergedReason reason;
    PetscReal rnorm;
    PetscInt its;
    ierr = KSPGetType(ksp, &ksptype); CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(ksp, &reason); CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp, &its); CHKERRQ(ierr);
    ierr = KSPGetResidualNorm(ksp, &rnorm); CHKERRQ(ierr);
    if (!test_mode || reason < 0 || rnorm > 1e-8) {
      ierr = PetscPrintf(comm, "KSP %s %s iterations %D rnorm %e\n", ksptype,
                         KSPConvergedReasons[reason], its, (double)rnorm); CHKERRQ(ierr);
    }
    if (benchmark_mode && !test_mode) {
      CeedInt gsize;
      ierr = VecGetSize(X, &gsize); CHKERRQ(ierr);
      MPI_Reduce(&my_rt, &rt_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
      MPI_Reduce(&my_rt, &rt_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
      ierr = PetscPrintf(comm,
                         "CG solve time  : %g (%g) sec.\n"
                         "DOFs/sec in CG : %g (%g) million.\n",
                         rt_max, rt_min,
                         1e-6*gsize*its/rt_max, 1e-6*gsize*its/rt_min);
      CHKERRQ(ierr);
    }
  }

 /* {
    PetscReal maxerror;
    ierr = ComputeErrorMax(user, op_error, X, target, &maxerror); CHKERRQ(ierr);
    if (!test_mode || maxerror > 5e-2) {
      ierr = PetscPrintf(comm, "Pointwise error (max) %e\n", (double)maxerror);
      CHKERRQ(ierr);
    }
  }*/

  ierr = VecDestroy(&rhs); CHKERRQ(ierr);
  ierr = VecDestroy(&rhsloc); CHKERRQ(ierr);
  ierr = VecDestroy(&X); CHKERRQ(ierr);
  ierr = VecDestroy(&user->Xloc); CHKERRQ(ierr);
  ierr = VecDestroy(&user->Yloc); CHKERRQ(ierr);
  ierr = MatDestroy(&mat); CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);

  CeedVectorDestroy(&user->xceed);
  CeedVectorDestroy(&user->yceed);
  CeedVectorDestroy(&user->rho);
  CeedVectorDestroy(&target);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_diff);
  CeedOperatorDestroy(&op_error);
  CeedElemRestrictionDestroy(&Erestrictu);
  CeedElemRestrictionDestroy(&Erestrictx);
  CeedElemRestrictionDestroy(&Erestrictui);
  CeedElemRestrictionDestroy(&Erestrictqdi);
  CeedElemRestrictionDestroy(&Erestrictxi);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_diff);
  CeedQFunctionDestroy(&qf_error);
  CeedBasisDestroy(&basisu);
  CeedBasisDestroy(&basisx);
  CeedDestroy(&ceed);
  ierr = PetscFree(user); CHKERRQ(ierr);
  return PetscFinalize();
}
