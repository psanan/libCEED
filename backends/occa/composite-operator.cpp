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

#include "composite-operator.hpp"


namespace ceed {
  namespace occa {
    CompositeOperator::CompositeOperator() :
        ceed(NULL) {}

    CompositeOperator::~CompositeOperator() {}

    CompositeOperator* CompositeOperator::from(CeedOperator op) {
      int ierr;
      CompositeOperator *operator_;
      CeedInt operatorCount;
      CeedOperator *subOperators;

      ierr = CeedOperatorGetData(op, (void**) &operator_); CeedOccaFromChk(ierr);
      ierr = CeedOperatorGetNumSub(op, &operatorCount); CeedOccaFromChk(ierr);
      ierr = CeedOperatorGetSubList(op, &subOperators); CeedOccaFromChk(ierr);

      operator_->ceedOperators.resize(operatorCount);
      for (int i = 0; i < operatorCount; ++i) {
        // TODO: Add NULL check
        operator_->ceedOperators[i] = Operator::from(subOperators[i]);
      }

      return operator_;
    }

    int CompositeOperator::apply(Vector &in, Vector &out, CeedRequest *request) {
      int ierr;
      const int operatorCount = (int) ceedOperators.size();

      for (int i = 0; i < operatorCount; ++i) {
        // TODO: Accumulate operators instead of overriding each one
        ierr = ceedOperators[i]->apply(in, out, request);
        CeedChk(ierr);
      }

      return 0;
    }

    //---[ Ceed Callbacks ]-----------
    int CompositeOperator::registerOperatorFunction(Ceed ceed, CeedOperator op,
                                           const char *fname, ceed::occa::ceedFunction f) {
      return CeedSetBackendFunction(ceed, "Operator", op, fname, f);
    }

    int CompositeOperator::ceedCreate(CeedOperator op) {

      int ierr;

      Ceed ceed;
      ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);

      CompositeOperator *operator_ = new CompositeOperator();
      ierr = CeedOperatorSetData(op, (void**) &operator_); CeedChk(ierr);

      ierr = registerOperatorFunction(ceed, op, "Apply",
                                      (ceed::occa::ceedFunction) CompositeOperator::ceedApply);
      CeedChk(ierr);

      ierr = registerOperatorFunction(ceed, op, "Destroy",
                                      (ceed::occa::ceedFunction) CompositeOperator::ceedDestroy);
      CeedChk(ierr);

      // TODO: Return 0 once the 2 issues above are fixed
      return CeedError(NULL, 1, "Backend does not implement composite operators");
    }

    int CompositeOperator::ceedApply(CeedOperator op,
                            CeedVector invec, CeedVector outvec, CeedRequest *request) {
      CompositeOperator *operator_ = CompositeOperator::from(op);
      Vector *in = Vector::from(invec);
      Vector *out = Vector::from(outvec);

      if (!operator_) {
        return CeedError(NULL, 1, "Incorrect CeedOperator argument: op");
      }
      if (!in) {
        return CeedError(operator_->ceed, 1, "Incorrect CeedVector argument: invec");
      }
      if (!out) {
        return CeedError(operator_->ceed, 1, "Incorrect CeedVector argument: outvec");
      }

      return operator_->apply(*in, *out, request);
    }

    int CompositeOperator::ceedDestroy(CeedOperator op) {
      delete CompositeOperator::from(op);
      return 0;
    }
  }
}
