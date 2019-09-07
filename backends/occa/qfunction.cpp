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

#include "qfunction.hpp"
#include "vector.hpp"


namespace ceed {
  namespace occa {
    QFunction::QFunction(::occa::device device,
                         const std::string &source) :
        ceed(NULL),
        ceedInputFields(0),
        ceedOutputFields(0),
        ceedContextSize(0),
        ceedContext(NULL) {
      const size_t colonIndex = source.find(':');
      filename = source.substr(0, colonIndex);
      kernelName = source.substr(colonIndex + 1);
    }

    QFunction::~QFunction() {
      qFunctionKernel.free();
      context.free();
    }

    QFunction* QFunction::from(CeedQFunction qf) {
      int ierr;
      QFunction *qFunction;

      ierr = CeedQFunctionGetData(qf, (void**) &qFunction); CeedOccaFromChk(ierr);
      ierr = CeedQFunctionGetCeed(qf, &qFunction->ceed); CeedOccaFromChk(ierr);
      ierr = CeedQFunctionGetNumArgs(
        qf,
        &qFunction->ceedInputFields,
        &qFunction->ceedOutputFields
      ); CeedOccaFromChk(ierr);

      ierr = CeedQFunctionGetContextSize(qf, &qFunction->ceedContextSize);
      CeedOccaFromChk(ierr);
      ierr = CeedQFunctionGetInnerContext(qf, (void**) &qFunction->context);
      CeedOccaFromChk(ierr);

      return qFunction;
    }

    ::occa::device QFunction::getDevice() {
      if (qFunctionKernel.isInitialized()) {
        return qFunctionKernel.getDevice();
      }
      return Context::from(ceed)->device;
    }

    int QFunction::buildKernel() {
      if (qFunctionKernel.isInitialized()) {
        return 0;
      }
      // TODO: Build kernel
      return 0;
    }

    int QFunction::syncContext() {
      if (ceedContextSize <= 0) {
        // TODO: context = occa::null;
        return 0;
      }

      if (ceedContextSize != (size_t) context.size()) {
        context.free();
        context = getDevice().malloc(ceedContextSize);
      }
      context.copyFrom(ceedContext);
      return 0;
    }

    int QFunction::apply(CeedInt Q, CeedVector *U, CeedVector *V) {
      int ierr;
      ierr = buildKernel(); CeedChk(ierr);
      ierr = syncContext(); CeedChk(ierr);

      QFunctionFields fields;
      for (CeedInt i = 0; i < ceedInputFields; i++) {
        Vector *u = Vector::from(U[i]);
        if (!u) {
          return CeedError(ceed, 1, "Incorrect qFunction input field: U[%i]", (int) i);
        }
        ierr = u->getArray(CEED_MEM_DEVICE, &fields.inputs[i]); CeedChk(ierr);
      }

      for (CeedInt i = 0; i < ceedOutputFields; i++) {
        Vector *v = Vector::from(V[i]);
        if (!v) {
          return CeedError(ceed, 1, "Incorrect qFunction output field: V[%i]", (int) i);
        }
        ierr = v->getArray(CEED_MEM_DEVICE, &fields.outputs[i]); CeedChk(ierr);
      }

      ::occa::kernelArg fieldsArg;
      fieldsArg.add(&fields, sizeof(QFunctionFields));
      qFunctionKernel(context, Q, fieldsArg);

      for (CeedInt i = 0; i < ceedInputFields; i++) {
        Vector *u = Vector::from(U[i]);
        ierr = u->restoreArray(&fields.inputs[i]); CeedChk(ierr);
      }

      for (CeedInt i = 0; i < ceedOutputFields; i++) {
        Vector *v = Vector::from(V[i]);
        ierr = v->restoreArray(&fields.outputs[i]); CeedChk(ierr);
      }

      return 0;
    }

    //---[ Ceed Callbacks ]-----------
    int QFunction::registerQFunctionFunction(Ceed ceed, CeedQFunction qf,
                                             const char *fname, ceed::occa::ceedFunction f) {
      return CeedSetBackendFunction(ceed, "QFunction", qf, fname, f);
    }

    int QFunction::ceedCreate(CeedQFunction qf) {
      // Based on cuda-gen
      int ierr;

      Ceed ceed;
      ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
      Context *context;
      ierr = CeedGetData(ceed, (void**) &context); CeedChk(ierr);
      char *source;
      ierr = CeedQFunctionGetSourcePath(qf, &source); CeedChk(ierr);

      QFunction *qFunction = new QFunction(context->device, source);
      ierr = CeedQFunctionSetData(qf, (void**) &qFunction); CeedChk(ierr);

      ierr = registerQFunctionFunction(ceed, qf, "Apply",
                                       (ceed::occa::ceedFunction) QFunction::ceedApply);
      CeedChk(ierr);

      ierr = registerQFunctionFunction(ceed, qf, "Destroy",
                                       (ceed::occa::ceedFunction) QFunction::ceedDestroy);
      CeedChk(ierr);

      return 0;
    }

    int QFunction::ceedApply(CeedQFunction qf, CeedInt Q,
                             CeedVector *U, CeedVector *V) {
      QFunction *qFunction = QFunction::from(qf);
      if (qFunction) {
        return qFunction->apply(Q, U, V);
      }

      return 1;
    }

    int QFunction::ceedDestroy(CeedQFunction qf) {
      delete QFunction::from(qf);
      return 0;
    }
  }
}
