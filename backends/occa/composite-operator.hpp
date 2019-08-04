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

#ifndef CEED_OCCA_COMPOSITEOPERATOR_HEADER
#define CEED_OCCA_COMPOSITEOPERATOR_HEADER

#include <vector>

#include "operator.hpp"


namespace ceed {
  namespace occa {
    typedef std::vector<ceed::occa::Operator*> OperatorVector_t;

    class CompositeOperator {
     public:
      // Ceed object information
      Ceed ceed;
      OperatorVector_t ceedOperators;

      CompositeOperator();

      ~CompositeOperator();

      static CompositeOperator* from(CeedOperator op);

      int apply(Vector &in, Vector &out, CeedRequest *request);

      //---[ Ceed Callbacks ]-----------
      static int registerOperatorFunction(Ceed ceed, CeedOperator op,
                                          const char *fname, ceed::occa::ceedFunction f);

      static int ceedCreate(CeedOperator op);

      static int ceedApply(CeedOperator op,
                           CeedVector invec, CeedVector outvec, CeedRequest *request);

      static int ceedDestroy(CeedOperator op);
    };
  }
}

#endif
