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
#include "operator.hpp"


namespace ceed {
  namespace occa {
    Operator::Operator() :
        ceed(NULL),
        ceedQ(0),
        ceedElementCount(0),
        ceedInputFieldCount(0),
        ceedOutputFieldCount(0),
        ceedOperatorInputFields(NULL),
        ceedOperatorOutputFields(NULL),
        ceedQFunctionInputFields(NULL),
        ceedQFunctionOutputFields(NULL),
        isInitialized(false) {}

    Operator::~Operator() {
      for (int i = 0; i < (int) eVectors.size(); ++i) {
        delete eVectors[i];
      }
      for (int i = 0; i < (int) qVectors.size(); ++i) {
        delete qVectors[i];
      }
    }

    Operator* Operator::from(CeedOperator op) {
      int ierr;
      Operator *operator_;
      CeedQFunction qf;

      ierr = CeedOperatorGetData(op, (void**) &operator_); CeedOccaFromChk(ierr);
      ierr = CeedOperatorGetQFunction(op, &qf); CeedOccaFromChk(ierr);

      // Get dimensions
      ierr = CeedOperatorGetNumQuadraturePoints(op, &operator_->ceedQ); CeedOccaFromChk(ierr);
      ierr = CeedOperatorGetNumElements(op, &operator_->ceedElementCount); CeedOccaFromChk(ierr);
      ierr = CeedQFunctionGetNumArgs(
        qf,
        &operator_->ceedInputFieldCount,
        &operator_->ceedOutputFieldCount
      ); CeedOccaFromChk(ierr);

      // Get Fields
      ierr = CeedOperatorGetFields(
        op,
        &operator_->ceedOperatorInputFields,
        &operator_->ceedOperatorOutputFields
      ); CeedOccaFromChk(ierr);
      ierr = CeedQFunctionGetFields(
        qf,
        &operator_->ceedQFunctionInputFields,
        &operator_->ceedQFunctionOutputFields
      ); CeedOccaFromChk(ierr);

      return operator_;
    }

    ::occa::device Operator::getDevice() {
      // if (qFunctionKernel.isInitialized()) {
      //   return qFunctionKernel.getDevice();
      // }
      return Context::from(ceed)->device;
    }

    int Operator::setup() {
      int ierr;

      const CeedInt fieldCount = ceedInputFieldCount + ceedOutputFieldCount;
      eVectors.resize(fieldCount);
      qVectors.resize(fieldCount);
      for (CeedInt i = 0; i < fieldCount; ++i) {
        eVectors[i] = new Vector();
        qVectors[i] = new Vector();
      }

      VectorVector_t eInputVectors(eVectors.begin(), eVectors.begin() + ceedInputFieldCount);
      VectorVector_t qInputVectors(qVectors.begin(), qVectors.begin() + ceedInputFieldCount);
      VectorVector_t eOutputVectors(eVectors.begin() + ceedInputFieldCount, eVectors.end());
      VectorVector_t qOutputVectors(qVectors.begin() + ceedInputFieldCount, qVectors.end());

      ierr = setupVectors(
        ceedInputFieldCount,
        eInputVectors,
        qInputVectors,
        ceedOperatorInputFields,
        ceedQFunctionInputFields
      ); CeedChk(ierr);

      ierr = setupVectors(
        ceedOutputFieldCount,
        eOutputVectors,
        qOutputVectors,
        ceedOperatorOutputFields,
        ceedQFunctionOutputFields
      ); CeedChk(ierr);

      return 0;
    }

    int Operator::setupVectors(CeedInt fieldCount,
                               VectorVector_t eVectors,
                               VectorVector_t qVectors,
                               CeedOperatorField *operatorFields,
                               CeedQFunctionField *qFunctionFields) {
      int ierr;
      for (CeedInt i = 0; i < fieldCount; ++i) {
        CeedOperatorField operatorField = operatorFields[i];
        CeedQFunctionField qFunctionField = qFunctionFields[i];
        Vector &qVector = *qVectors[i];
        Vector &eVector = *eVectors[i];

        CeedEvalMode emode;
        ierr = CeedQFunctionFieldGetEvalMode(qFunctionField, &emode); CeedChk(ierr);

        // Get component count
        CeedInt componentCount = 0;
        switch (emode) {
          case CEED_EVAL_NONE:
          case CEED_EVAL_INTERP:
          case CEED_EVAL_GRAD:
            ierr = CeedQFunctionFieldGetNumComponents(qFunctionField, &componentCount); CeedChk(ierr);
            break;
          case CEED_EVAL_WEIGHT:
            componentCount = 1;
            break;
          case CEED_EVAL_DIV:
            // TODO: Not implemented
            return CeedError(ceed, 1, "CEED_EVAL_DIV is not implemented yet");
          case CEED_EVAL_CURL:
            // TODO: Not implemented
            return CeedError(ceed, 1, "CEED_EVAL_CURL is not implemented yet");
          default:
            return CeedError(ceed, 1, "QFunctionField EvalMode is not implemented yet");
        }

        // Get basis
        Basis *basis = NULL;
        switch (emode) {
          case CEED_EVAL_GRAD:
          case CEED_EVAL_WEIGHT:
            basis = Basis::from(operatorField);
            if (!basis) {
              return CeedError(ceed, 1, "Incorrect CeedBasis from opfield[%i]", (int) i);
            }
            break;
          default: {}
        }

        // Resize qVector
        switch (emode) {
          case CEED_EVAL_NONE:
          case CEED_EVAL_INTERP:
          case CEED_EVAL_WEIGHT:
            qVector.resize(ceedElementCount * ceedQ * componentCount);
            break;
          case CEED_EVAL_GRAD:
            qVector.resize(ceedElementCount * ceedQ * componentCount * basis->ceedDim);
            break;
          default:
            return CeedError(ceed, 1, "QFunctionField EvalMode is not implemented yet");
        }

        // Apply weight
        if (emode == CEED_EVAL_WEIGHT) {
          ierr = basis->apply(
            ceedElementCount,
            CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT,
            NULL, &qVector
          ); CeedChk(ierr);
        }

#if 0
        if (emode != CEED_EVAL_WEIGHT) {
          ElemRestriction *restrict = ElemRestriction::from(operatorField);
          if (!restrict) {
            return CeedError(ceed, 1, "Incorrect ElemRestriction from opfield[%i]", (int) i);
          }
          ierr = restrict->setupVector(NULL, eVector); CeedChk(ierr);
        }
#endif
      }
      return 0;
    }

    int Operator::apply(Vector &in, Vector &out, CeedRequest *request) {
      if (!isInitialized) {
        setup();
      }
      // TODO: Implement
      return 0;
    }

    //---[ Ceed Callbacks ]-----------
    int Operator::registerOperatorFunction(Ceed ceed, CeedOperator op,
                                           const char *fname, ceed::occa::ceedFunction f) {
      return CeedSetBackendFunction(ceed, "Operator", op, fname, f);
    }

    int Operator::ceedCreate(CeedOperator op) {
      // Based on cuda-gen
      int ierr;

      Ceed ceed;
      ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);

      Operator *operator_ = new Operator();
      ierr = CeedOperatorSetData(op, (void**) &operator_); CeedChk(ierr);

      ierr = registerOperatorFunction(ceed, op, "Apply",
                                      (ceed::occa::ceedFunction) Operator::ceedApply);
      CeedChk(ierr);

      ierr = registerOperatorFunction(ceed, op, "Destroy",
                                      (ceed::occa::ceedFunction) Operator::ceedDestroy);
      CeedChk(ierr);

      return 0;
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
