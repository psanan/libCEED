// Copyright (c) 2019, Lawrence Livermore National Security, LLC.
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

#include "tensor-basis.hpp"
#include "kernels/tensor-basis-1d.okl"
#include "kernels/tensor-basis-2d.okl"
#include "kernels/tensor-basis-3d.okl"


namespace ceed {
  namespace occa {
    TensorBasis::TensorBasis() :
        isInitialized(false) {}

    TensorBasis::~TensorBasis() {
      interpKernel.free();
      gradKernel.free();
      weightKernel.free();
      interp1D.free();
      grad1D.free();
      qWeight1D.free();
      interpWeights.free();
      interpGradWeights.free();
    }

    int TensorBasis::setup() {
      if (isInitialized) {
        return 0;
      }

      isInitialized = true;
      return 0;
    }

    int TensorBasis::apply(const CeedInt elementCount,
                           CeedTransposeMode tmode,
                           CeedEvalMode emode,
                           Vector *u,
                           Vector *v) {
      setup();

      return 0;
    }

    //---[ Ceed Callbacks ]-------------
    int TensorBasis::ceedCreate(CeedInt dim,
                                CeedInt P1D, CeedInt Q1D,
                                const CeedScalar *interp1D,
                                const CeedScalar *grad1D,
                                const CeedScalar *qref1D,
                                const CeedScalar *qweight1D,
                                CeedBasis basis) {
      // Based on cuda-shared
      if (Q1D < P1D) {
        return CeedError(NULL, 1, "Backend does not implement underintegrated basis.");
      }

      int ierr;
      Ceed ceed;
      ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);

      TensorBasis *basis_ = new TensorBasis();
      ierr = CeedBasisSetData(basis, (void**) &basis_); CeedChk(ierr);

      ierr = registerBasisFunction(ceed, basis, "Apply",
                                   (ceed::occa::ceedFunction) Basis::ceedApply);
      CeedChk(ierr);

      ierr = registerBasisFunction(ceed, basis, "Destroy",
                                   (ceed::occa::ceedFunction) Basis::ceedDestroy);
      CeedChk(ierr);

      return 0;
    }
  }
}
