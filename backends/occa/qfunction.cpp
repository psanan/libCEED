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

#include <sstream>

#include "qfunction.hpp"
#include "vector.hpp"


namespace ceed {
  namespace occa {
    QFunction::QFunction(::occa::device device,
                         const std::string &source) :
        ceed(NULL),
        ceedInputFields(0),
        ceedOutputFields(0) {
      OCCA_DEBUG_TRACE("qfunction: QFunction");
      const size_t colonIndex = source.find(':');
      filename = source.substr(0, colonIndex);
      qFunctionName = source.substr(colonIndex + 1);
    }

    QFunction::~QFunction() {
      OCCA_DEBUG_TRACE("qfunction: ~QFunction");
      qFunctionKernel.free();
    }

    QFunction* QFunction::from(CeedQFunction qf) {
      OCCA_DEBUG_TRACE("qfunction: from");
      int ierr;
      QFunction *qFunction;

      ierr = CeedQFunctionGetData(qf, (void**) &qFunction); CeedOccaFromChk(ierr);
      ierr = CeedQFunctionGetCeed(qf, &qFunction->ceed); CeedOccaFromChk(ierr);
      ierr = CeedQFunctionGetNumArgs(
        qf,
        &qFunction->ceedInputFields,
        &qFunction->ceedOutputFields
      ); CeedOccaFromChk(ierr);

      CeedQFunctionField *inputFields, *outputFields;
      CeedInt size;
      ierr = CeedQFunctionGetFields(qf, &inputFields, &outputFields);
      CeedOccaFromChk(ierr);

      for (int i = 0; i < qFunction->ceedInputFields; ++i) {
        ierr = CeedQFunctionFieldGetSize(inputFields[i], &size); CeedOccaFromChk(ierr);
        qFunction->ceedInputFieldSizes.push_back(size);
      }
      for (int i = 0; i < qFunction->ceedOutputFields; ++i) {
        ierr = CeedQFunctionFieldGetSize(outputFields[i], &size); CeedOccaFromChk(ierr);
        qFunction->ceedOutputFieldSizes.push_back(size);
      }

      // TODO: Add context back if needed
      // ierr = CeedQFunctionGetContextSize(qf, &qFunction->ceedContextSize);
      // CeedOccaFromChk(ierr);
      // ierr = CeedQFunctionGetInnerContext(qf, (void**) &qFunction->context);
      // CeedOccaFromChk(ierr);

      return qFunction;
    }

    ::occa::device QFunction::getDevice() {
      OCCA_DEBUG_TRACE("qfunction: getDevice");
      if (qFunctionKernel.isInitialized()) {
        return qFunctionKernel.getDevice();
      }
      return Context::from(ceed)->device;
    }

    int QFunction::buildKernel(const CeedInt Q) {
      OCCA_DEBUG_TRACE("qfunction: buildKernel");

      // TODO: Store a kernel per Q
      if (qFunctionKernel.isInitialized()) {
        return 0;
      }

      const std::string kernelName = "qFunctionKernel";

      ::occa::properties props;
      props["defines/CeedInt"] = ::occa::dtype::get<CeedInt>().name();
      props["defines/CeedScalar"] = ::occa::dtype::get<CeedScalar>().name();

      std::stringstream ss;
      ss << "#define CEED_QFUNCTION(FUNC_NAME) \\" << std::endl
         << "  inline int FUNC_NAME"               << std::endl
         <<                                           std::endl
         << "#include \"" << filename << "\""      << std::endl
         <<                                           std::endl
         << getKernelSource(kernelName, Q)         << std::endl;

      qFunctionKernel = (
        getDevice().buildKernelFromString(ss.str(),
                                          kernelName,
                                          props)
      );

      return 0;
    }

    std::string QFunction::getKernelSource(const std::string &kernelName,
                                           const CeedInt Q) {
      OCCA_DEBUG_TRACE("qfunction: getKernelSource");

      const int lastArg = ceedInputFields + ceedOutputFields - 1;
      std::stringstream ss;

      ss << "@kernel void " << kernelName << "("                                        << std::endl;

      // qfunction arguments
      for (int i = 0; i < ceedInputFields; ++i) {
        const char end = (i != lastArg) ? ',' : ' ';
        ss << "  const CeedScalar *in_" << i << end                                     << std::endl;
      }
      for (int i = 0; i < ceedOutputFields; ++i) {
        const char end = (i != lastArg) ? ',' : ' ';
        ss << "  CeedScalar *out_" << i << end                                          << std::endl;
      }

      // qfunction body
      ss << ") {"                                                                       << std::endl;

      // Call the real qfunction
      ss << "  for (int q = 0; q < " << Q << "; ++q; @tile(128, @outer, @inner)) {"     << std::endl
         << "    const CeedScalar* in[" << std::max(ceedInputFields, 1) << "];"         << std::endl
         << "    CeedScalar* out[" << std::max(ceedOutputFields, 1) << "];"             << std::endl;

      // Set and define in for the q point
      for (int i = 0; i < ceedInputFields; ++i) {
        const CeedInt fieldSize = ceedInputFieldSizes[i];
        const std::string qIn_i = "qIn_" + ::occa::toString(i);
        const std::string in_i = "in_" + ::occa::toString(i);

        ss << "    CeedScalar " << qIn_i << "[" << fieldSize << "];"                    << std::endl
           << "    in[" << i << "] = " << qIn_i << ";"                                  << std::endl
            // Copy q data
           << "    for (int q_i = 0; q_i < " << fieldSize << "; ++q_i) {"               << std::endl
           << "      " << qIn_i << "[q_i] = " << in_i << "[q + (" << Q << " * q_i)];"   << std::endl
           << "    }"                                                                   << std::endl;
      }

      // Set out for the q point
      for (int i = 0; i < ceedOutputFields; ++i) {
        const CeedInt fieldSize = ceedOutputFieldSizes[i];
        const std::string qOut_i = "qOut_" + ::occa::toString(i);

        ss << "    CeedScalar " << qOut_i << "[" << fieldSize << "];"                   << std::endl
           << "    out[" << i << "] = " << qOut_i << ";"                                << std::endl;
      }

      ss << "    " << qFunctionName << "(NULL, 1, in, out);"                            << std::endl;

      // Copy out for the q point
      for (int i = 0; i < ceedOutputFields; ++i) {
        const CeedInt fieldSize = ceedOutputFieldSizes[i];
        const std::string qOut_i = "qOut_" + ::occa::toString(i);
        const std::string out_i = "out_" + ::occa::toString(i);

        ss << "    for (int q_i = 0; q_i < " << fieldSize << "; ++q_i) {"               << std::endl
           << "      " << out_i << "[q + (" << Q << " * q_i)] = " << qOut_i << "[q_i];" << std::endl
           << "    }"                                                                   << std::endl;
      }

      ss << "  }"                                                                       << std::endl
         << "}";

      return ss.str();
    }

    int QFunction::apply(CeedInt Q, CeedVector *U, CeedVector *V) {
      OCCA_DEBUG_TRACE("qfunction: apply");
      int ierr;
      ierr = buildKernel(Q); CeedChk(ierr);

      // TODO: Add context back if needed
      // ierr = syncContext(); CeedChk(ierr);

      std::vector<CeedScalar*> inputArgs, outputArgs;

      qFunctionKernel.clearArgs();
      for (CeedInt i = 0; i < ceedInputFields; i++) {
        Vector *u = Vector::from(U[i]);
        if (!u) {
          return CeedError(ceed, 1, "Incorrect qFunction input field: U[%i]", (int) i);
        }
        CeedScalar *inputArg;
        ierr = u->getArray(CEED_MEM_DEVICE, &inputArg); CeedChk(ierr);

        inputArgs.push_back(inputArg);
        qFunctionKernel.pushArg(arrayToMemory(inputArg));
      }

      for (CeedInt i = 0; i < ceedOutputFields; i++) {
        Vector *v = Vector::from(V[i]);
        if (!v) {
          return CeedError(ceed, 1, "Incorrect qFunction output field: V[%i]", (int) i);
        }
        CeedScalar *outputArg;
        ierr = v->getArray(CEED_MEM_DEVICE, &outputArg); CeedChk(ierr);

        outputArgs.push_back(outputArg);
        qFunctionKernel.pushArg(arrayToMemory(outputArg));
      }

      qFunctionKernel.run();

      for (CeedInt i = 0; i < ceedInputFields; i++) {
        Vector *u = Vector::from(U[i]);
        ierr = u->restoreArray(&inputArgs[i]); CeedChk(ierr);
      }

      for (CeedInt i = 0; i < ceedOutputFields; i++) {
        Vector *v = Vector::from(V[i]);
        ierr = v->restoreArray(&outputArgs[i]); CeedChk(ierr);
      }

      return 0;
    }

    //---[ Ceed Callbacks ]-----------
    int QFunction::registerQFunctionFunction(Ceed ceed, CeedQFunction qf,
                                             const char *fname, ceed::occa::ceedFunction f) {
      OCCA_DEBUG_TRACE("qfunction: registerQFunctionFunction");
      return CeedSetBackendFunction(ceed, "QFunction", qf, fname, f);
    }

    int QFunction::ceedCreate(CeedQFunction qf) {
      OCCA_DEBUG_TRACE("qfunction: ceedCreate");
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
      OCCA_DEBUG_TRACE("qfunction: ceedApply");
      QFunction *qFunction = QFunction::from(qf);
      if (qFunction) {
        OCCA_DEBUG_TRACE("qfunction: from");
        return qFunction->apply(Q, U, V);
      }

      return 1;
    }

    int QFunction::ceedDestroy(CeedQFunction qf) {
      OCCA_DEBUG_TRACE("qfunction: ceedDestroy");
      delete QFunction::from(qf);
      return 0;
    }
  }
}
