// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "ceed-xsmm.h"

//------------------------------------------------------------------------------
// Get Kernel Index
//------------------------------------------------------------------------------
int CeedGetXsmmInd_Tensor(CeedInt nelem, CeedInt add, CeedTransposeMode tmode,
                          CeedInt B, CeedInt C, CeedInt J, CeedInt currdim,
                          CeedInt dim) {
  return (nelem == 8 ? 1:0)*4*2*dim + (add ? 1:0)*4*dim +
         (tmode ? 1:0)*2*dim + (B == J ? 1:0)*dim + currdim;
}

int CeedGetXsmmInd_NonTensor(CeedInt add, CeedInt P, CeedInt Q, CeedInt B,
                             CeedInt C, CeedInt J) {
  return (C == 8 ? 1:0)*4*2 + (add ? 1:0)*4 +
         (B == P ? (J == Q ? 0:1) : (B == Q ? 2:3));
}

//------------------------------------------------------------------------------
// Tensor Contract Apply
//------------------------------------------------------------------------------
static int CeedTensorContractApply_Xsmm(CeedTensorContract contract, CeedInt A,
                                        CeedInt B, CeedInt C, CeedInt J,
                                        const CeedScalar *restrict t,
                                        CeedTransposeMode tmode,
                                        const CeedInt add,
                                        const CeedScalar *restrict u,
                                        CeedScalar *restrict v) {
  int ierr;
  CeedInt blksize = 8, ind, nelem;
  CeedTensorContract_Xsmm *impl;
  ierr = CeedTensorContractGetData(contract, (void *)&impl); CeedChk(ierr);

  // Get nelem and current dim
  CeedScalar currdim = log(C/blksize) / log(J);
  if (!(C % blksize) && currdim - (int)currdim < 1e-15) {
    nelem = blksize;
  } else {
    nelem = 1;
    currdim = log(C) / log(J);
  }

  // Get kernel index
  if (impl->tensorbasis)
    ind = CeedGetXsmmInd_Tensor(nelem, add, tmode==CEED_TRANSPOSE?1:0, B, C, J,
                                (CeedInt)currdim, impl->dim);
  else
    ind = CeedGetXsmmInd_NonTensor(add, impl->P, impl->Q, B, C, J);

  // Run kernel
  impl->kernels[ind](u, t, v);

  return 0;
}

//------------------------------------------------------------------------------
// Tensor Contract Destroy
//------------------------------------------------------------------------------
static int CeedTensorContractDestroy_Xsmm(CeedTensorContract contract) {
  int ierr;
  CeedTensorContract_Xsmm *impl;
  ierr = CeedTensorContractGetData(contract, (void *)&impl); CeedChk(ierr);
  // Free kernels
  if (impl->tensorbasis) {
    for (CeedInt nelem = 1; nelem <= 8; nelem+=7)
      for (CeedInt add = 0; add <= 1; add++)
        for (CeedInt tmode = 0; tmode <= 1; tmode++)
          for (CeedInt grad = 0; grad <=1; grad++)
            for (CeedInt dim = 0; dim < impl->dim; dim++) {
              CeedInt B = grad ? impl->Q : (tmode ? impl->Q : impl->P),
                      J = grad ? impl->Q : (tmode ? impl->P : impl->Q),
                      C = nelem*CeedIntPow(J, dim);
              int ind = CeedGetXsmmInd_Tensor(nelem, add, tmode, B, C, J, dim,
                                              impl->dim);
              libxsmm_release_kernel(&impl->kernels[ind]);
            }
  } else {
    for (CeedInt nelem = 1; nelem <= 8; nelem+=7)
      for (CeedInt add = 0; add <= 1; add++)
        for (CeedInt tmode = 0; tmode <= 1; tmode++)
          for (CeedInt grad = 1; grad <= impl->dim; grad+=impl->dim-1) {
            CeedInt B = tmode ? grad*impl->Q : impl->P,
                    J = tmode ? impl->P : grad*impl->Q;
            int ind = CeedGetXsmmInd_NonTensor(add, impl->P, impl->Q, B, nelem,
                                               J);
            libxsmm_release_kernel(&impl->kernels[ind]);
          }
  }
  ierr = CeedFree(&impl->kernels); CeedChk(ierr);
  ierr = CeedFree(&impl); CeedChk(ierr);

  return 0;
}

//------------------------------------------------------------------------------
// Tensor Contract Create
//------------------------------------------------------------------------------
int CeedTensorContractCreate_Xsmm(CeedBasis basis,
                                  CeedTensorContract contract) {
  int ierr;
  Ceed ceed;
  ierr = CeedTensorContractGetCeed(contract, &ceed); CeedChk(ierr);
  CeedTensorContract_Xsmm *impl;
  ierr = CeedCalloc(1, &impl); CeedChk(ierr);

  // Set up pointers to kernels
  ierr = CeedBasisGetTensorStatus(basis, &impl->tensorbasis); CeedChk(ierr);
  if (impl->tensorbasis) {
    ierr = CeedBasisGetNumNodes1D(basis, &impl->P); CeedChk(ierr);
    ierr = CeedBasisGetNumQuadraturePoints1D(basis, &impl->Q); CeedChk(ierr);
    ierr = CeedBasisGetDimension(basis, &impl->dim); CeedChk(ierr);
    ierr = CeedBasisGetNumComponents(basis, &impl->ncomp); CeedChk(ierr);
    // Set up kernel pointer array
    impl->numkernels = 2*2*4*impl->dim;
    ierr = CeedCalloc(impl->numkernels, &impl->kernels); CeedChk(ierr);
    for (CeedInt nelem = 1; nelem <= 8; nelem+=7)
      for (CeedInt add = 0; add <= 1; add++)
        for (CeedInt tmode = 0; tmode <= 1; tmode++)
          for (CeedInt grad = 0; grad <=1; grad++)
            for (CeedInt dim = 0; dim < impl->dim; dim++) {
              // Kernel options
              const int flags = LIBXSMM_GEMM_FLAGS('N', tmode ? 'T' : 'N');
              CeedInt B = grad ? impl->Q : (tmode ? impl->Q : impl->P),
                      J = grad ? impl->Q : (tmode ? impl->P : impl->Q),
                      C = nelem*CeedIntPow(J, dim),
                      A = impl->ncomp*CeedIntPow(B, impl->dim-1 - dim);
              int ind = CeedGetXsmmInd_Tensor(nelem, add, tmode, B, C, J, dim,
                                              impl->dim);
              CeedScalar alpha = 1.0, beta = 1.0;
              if (!add)
                beta = 0.0;

              // LIBXSMM kernel
              libxsmm_descriptor_blob xgemm_blob;
              const libxsmm_gemm_prefetch_type prefetch =
                                                 LIBXSMM_GEMM_PREFETCH_NONE;
              const libxsmm_gemm_descriptor* xgemm_desc = 0;
              xgemm_desc = libxsmm_gemm_descriptor_dinit(&xgemm_blob,
                             LIBXSMM_GEMM_PRECISION(double),
                             C, J, B, B, J, J, alpha, beta, flags, prefetch);
              impl->kernels[ind] = libxsmm_create_pgemm_ac_rm(xgemm_desc, A).dmm;
              if (!impl->kernels[ind])
                // LCOV_EXCL_START
                return CeedError(ceed, 1, "LIBXSMM kernel failed to build.");
              // LCOV_EXCL_STOP
            }
  } else {
    ierr = CeedBasisGetNumNodes(basis, &impl->P); CeedChk(ierr);
    ierr = CeedBasisGetNumQuadraturePoints(basis, &impl->Q); CeedChk(ierr);
    ierr = CeedBasisGetDimension(basis, &impl->dim); CeedChk(ierr);
    ierr = CeedBasisGetNumComponents(basis, &impl->ncomp); CeedChk(ierr);
    // Set up kernel pointer array
    impl->numkernels = 4*2*2;
    ierr = CeedCalloc(impl->numkernels, &impl->kernels); CeedChk(ierr);
    for (CeedInt nelem = 1; nelem <= 8; nelem+=7)
      for (CeedInt add = 0; add <= 1; add++)
        for (CeedInt tmode = 0; tmode <= 1; tmode++)
          for (CeedInt grad = 1; grad <= impl->dim; grad+=impl->dim-1) {
            const int flags = LIBXSMM_GEMM_FLAGS('N', tmode ? 'T' : 'N');
            CeedInt B = tmode ? grad*impl->Q : impl->P,
                    J = tmode ? impl->P : grad*impl->Q,
                    C = nelem, A = impl->ncomp;
            int ind = CeedGetXsmmInd_NonTensor(add, impl->P, impl->Q, B, C, J);
            CeedScalar alpha = 1.0, beta = 1.0;
            if (!add)
              beta = 0.0;

            // LIBXSMM kernel
            libxsmm_descriptor_blob xgemm_blob;
            const libxsmm_gemm_prefetch_type prefetch =
                                               LIBXSMM_GEMM_PREFETCH_NONE;
            const libxsmm_gemm_descriptor* xgemm_desc = 0;
            xgemm_desc = libxsmm_gemm_descriptor_dinit(&xgemm_blob,
                           LIBXSMM_GEMM_PRECISION(double),
                           C, J, B, B, J, J, alpha, beta, flags, prefetch);
            impl->kernels[ind] = libxsmm_create_pgemm_ac_rm(xgemm_desc, A).dmm;
            if (!impl->kernels[ind])
              // LCOV_EXCL_START
              return CeedError(ceed, 1, "LIBXSMM kernel failed to build.");
            // LCOV_EXCL_STOP
          }
  }
  ierr = CeedTensorContractSetData(contract, (void *)&impl); CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "TensorContract", contract, "Apply",
                                CeedTensorContractApply_Xsmm); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "TensorContract", contract, "Destroy",
                                CeedTensorContractDestroy_Xsmm); CeedChk(ierr);

  return 0;
}
//------------------------------------------------------------------------------
