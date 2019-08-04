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

#ifndef CEED_OCCA_BASIS_HEADER
#define CEED_OCCA_BASIS_HEADER

#include "types.hpp"
#include "vector.hpp"


namespace ceed {
  namespace occa {
    class Basis {
     public:
      // Ceed object information
      Ceed ceed;
      CeedInt ceedDim;
      CeedInt ceedQuadraturePoints;
      CeedInt ceedNodes;
      CeedInt ceedComponents;

      // Owned resources
      ::occa::kernel interpKernel;
      ::occa::kernel gradKernel;
      ::occa::kernel weightKernel;
      ::occa::memory interp;
      ::occa::memory grad;
      ::occa::memory qWeight;

      Basis();

      virtual ~Basis();

      static Basis* from(CeedBasis basis);
      static Basis* from(CeedOperatorField operatorField);

      ::occa::device getDevice();

      virtual int apply(const CeedInt elements,
                        CeedTransposeMode tmode,
                        CeedEvalMode emode,
                        Vector *u,
                        Vector *v) = 0;

      //---[ Ceed Callbacks ]-----------
      static int registerBasisFunction(Ceed ceed, CeedBasis basis,
                                       const char *fname, ceed::occa::ceedFunction f);

      template <class BasisClass>
      static int ceedCreate(CeedBasis basis) {
        int ierr;

        Ceed ceed;
        ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);

        Basis *basis_ = new BasisClass();
        ierr = CeedBasisSetData(basis, (void**) &basis_); CeedChk(ierr);

        ierr = registerBasisFunction(ceed, basis, "Apply",
                                     (ceed::occa::ceedFunction) Basis::ceedApply);
        CeedChk(ierr);

        ierr = registerBasisFunction(ceed, basis, "Destroy",
                                     (ceed::occa::ceedFunction) Basis::ceedDestroy);
        CeedChk(ierr);

        return 0;
      }

      static int ceedApply(CeedBasis basis, const CeedInt nelem,
                           CeedTransposeMode tmode,
                           CeedEvalMode emode, CeedVector u, CeedVector v);

      static int ceedDestroy(CeedBasis basis);
    };
  }
}

#endif
