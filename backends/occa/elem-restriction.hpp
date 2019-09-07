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

#ifndef CEED_OCCA_ELEMRESTRICTION_HEADER
#define CEED_OCCA_ELEMRESTRICTION_HEADER

#include "types.hpp"
#include "vector.hpp"


namespace ceed {
  namespace occa {
    class ElemRestriction {
     public:
      // Ceed object information
      Ceed ceed;
      CeedInt ceedElementCount;
      CeedInt ceedElementSize;
      CeedInt ceedComponentCount;
      CeedInt ceedBlockSize;

      // Owned resources

      ElemRestriction();

      ~ElemRestriction();

      static ElemRestriction* from(CeedElemRestriction r);
      static ElemRestriction* from(CeedOperatorField operatorField);

      ::occa::device getDevice();

      int apply(CeedTransposeMode tmode, CeedTransposeMode lmode,
                Vector &u, Vector &v, CeedRequest *request);

      int applyBlock(CeedInt block,
                     CeedTransposeMode tmode, CeedTransposeMode lmode,
                     Vector &u, Vector &v, CeedRequest *request);

      //---[ Ceed Callbacks ]-----------
      static int registerRestrictionFunction(Ceed ceed, CeedElemRestriction r,
                                             const char *fname, ceed::occa::ceedFunction f);

      static int ceedCreate(CeedElemRestriction r);

      static int ceedApply(CeedElemRestriction r,
                           CeedTransposeMode tmode, CeedTransposeMode lmode,
                           CeedVector u, CeedVector v, CeedRequest *request);

      static int ceedApplyBlock(CeedElemRestriction r,
                                CeedInt block,
                                CeedTransposeMode tmode, CeedTransposeMode lmode,
                                CeedVector u, CeedVector v, CeedRequest *request);

      static int ceedDestroy(CeedElemRestriction r);
    };
  }
}

#endif
