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

#include "basis.hpp"
#include "elem-restriction.hpp"
#include "operator-kernel-builder.hpp"
#include "operator.hpp"
#include "cpu-operator.hpp"
#include "gpu-operator.hpp"
#include "qfunction.hpp"


namespace ceed {
  namespace occa {
    Operator::Operator() :
        ceedQ(0),
        ceedElementCount(0),
        qfunction(NULL),
        needsInitialSetup(true) {}

    Operator::~Operator() {}

    Operator* Operator::from(CeedOperator op) {
      if (!op) {
        return NULL;
      }

      int ierr;
      Operator *operator_;

      ierr = CeedOperatorGetData(op, (void**) &operator_); CeedOccaFromChk(ierr);
      ierr = CeedOperatorGetCeed(op, &operator_->ceed); CeedOccaFromChk(ierr);

      operator_->qfunction = QFunction::from(op);
      if (!operator_->qfunction) {
        return NULL;
      }

      ierr = CeedOperatorGetNumQuadraturePoints(op, &operator_->ceedQ); CeedOccaFromChk(ierr);
      ierr = CeedOperatorGetNumElements(op, &operator_->ceedElementCount); CeedOccaFromChk(ierr);

      operator_->args.setupArgs(op);
      if (!operator_->args.isValid()) {
        return NULL;
      }

      return operator_;
    }

    int Operator::apply(Vector &in, Vector &out, CeedRequest *request) {
      if (!applyKernel.isInitialized()) {
        applyKernel = buildApplyKernel();

        if (!applyKernel.isInitialized()) {
          return CeedError(NULL, 1, "Error building apply kernel");
        }
      }

      if (needsInitialSetup) {
        initialSetup();
        needsInitialSetup = false;
      }

      apply(in, out);

      return 0;
    }

    //---[ Virtual Methods ]------------
    void Operator::initialSetup() {}

    //---[ Ceed Callbacks ]-------------
    int Operator::registerOperatorFunction(Ceed ceed, CeedOperator op,
                                           const char *fname, ceed::occa::ceedFunction f) {
      return CeedSetBackendFunction(ceed, "Operator", op, fname, f);
    }

    int Operator::ceedCreate(CeedOperator op) {
      int ierr;
      Ceed ceed;
      ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);

      Operator *operator_ = (
        (Context::from(ceed)->usingCpuDevice())
        ? ((Operator*) new CpuOperator())
        : ((Operator*) new CpuOperator()) // TODO: Use GpuOperator
      );

      ierr = CeedOperatorSetData(op, (void**) &operator_); CeedChk(ierr);

      ierr = registerOperatorFunction(ceed, op, "AssembleLinearQFunction",
                                      (ceed::occa::ceedFunction) Operator::ceedAssembleLinearQFunction);
      CeedChk(ierr);

      ierr = registerOperatorFunction(ceed, op, "Apply",
                                      (ceed::occa::ceedFunction) Operator::ceedApply);
      CeedChk(ierr);

      ierr = registerOperatorFunction(ceed, op, "Destroy",
                                      (ceed::occa::ceedFunction) Operator::ceedDestroy);
      CeedChk(ierr);

      return 0;
    }

    int Operator::ceedAssembleLinearQFunction(CeedOperator op) {
      return CeedError(NULL, 1, "Backend does not implement AssembleLinearQFunction");
    }

    int Operator::ceedApply(CeedOperator op,
                            CeedVector invec, CeedVector outvec, CeedRequest *request) {
      Operator *operator_ = Operator::from(op);
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

    int Operator::ceedDestroy(CeedOperator op) {
      delete Operator::from(op);
      return 0;
    }
  }
}