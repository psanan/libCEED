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

#ifndef CEED_OCCA_TENSORBASIS_HEADER
#define CEED_OCCA_TENSORBASIS_HEADER

#include "basis.hpp"


namespace ceed {
  namespace occa {
    class TensorBasis : public Basis {
     public:
      bool isInitialized;
      ::occa::kernel interpKernel;
      ::occa::kernel gradKernel;
      ::occa::kernel weightKernel;
      ::occa::memory interp1D;
      ::occa::memory grad1D;
      ::occa::memory qWeight1D;
      ::occa::memory interpWeights;
      ::occa::memory interpGradWeights;

      TensorBasis();

      ~TensorBasis();

      int setup();

      int apply(const CeedInt elementCount,
                CeedTransposeMode tmode,
                CeedEvalMode emode,
                Vector *u,
                Vector *v);

      //---[ Ceed Callbacks ]-----------
      static int ceedCreate(CeedInt dim,
                            CeedInt P1d, CeedInt Q1d,
                            const CeedScalar *interp1d,
                            const CeedScalar *grad1d,
                            const CeedScalar *qref1d,
                            const CeedScalar *qweight1d,
                            CeedBasis basis);
    };
  }
}

#endif
