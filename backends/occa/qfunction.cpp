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
    QFunction::QFunction(const std::string &source) {
      const size_t colonIndex = source.find(':');
      filename = source.substr(0, colonIndex);
      qFunctionName = source.substr(colonIndex + 1);
    }

    QFunction::~QFunction() {
      qFunctionKernel.free();
    }

    QFunction* QFunction::from(CeedQFunction qf) {
      if (!qf) {
        return NULL;
      }

      int ierr;
      QFunction *qFunction;

      ierr = CeedQFunctionGetData(qf, (void**) &qFunction);
      CeedOccaFromChk(ierr);

      ierr = CeedQFunctionGetCeed(qf, &qFunction->ceed);
      CeedOccaFromChk(ierr);

      ierr = CeedQFunctionGetContextSize(qf, &qFunction->ceedContextSize);
      CeedOccaFromChk(ierr);

      ierr = CeedQFunctionGetInnerContext(qf, &qFunction->ceedContext);
      CeedOccaFromChk(ierr);

      qFunction->args.setupQFunctionArgs(qf);
      if (!qFunction->args.isValid()) {
        return NULL;
      }

      return qFunction;
    }

    int QFunction::buildKernel(const CeedInt Q) {
      // TODO: Store a kernel per Q
      if (qFunctionKernel.isInitialized()) {
        return 0;
      }

      const std::string kernelName = "qFunctionKernel";

      ::occa::properties props;
      props["defines/CeedInt"] = ::occa::dtype::get<CeedInt>().name();
      props["defines/CeedScalar"] = ::occa::dtype::get<CeedScalar>().name();
      props["defines/CeedPragmaSIMD"] = "";

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
      std::stringstream ss;

      ss << "@kernel void " << kernelName << "("                                        << std::endl;

      // qfunction arguments
      for (int i = 0; i < args.inputCount(); ++i) {
        ss << "  const CeedScalar *in_" << i << ','                                     << std::endl;
      }
      for (int i = 0; i < args.outputCount(); ++i) {
        ss << "  CeedScalar *out_" << i << ','                                          << std::endl;
      }
      ss << "  void *ctx"                                                               << std::endl;
      ss << ") {"                                                                       << std::endl;

      // qfunction body

      // Call the real qfunction
      ss << "  for (int q = 0; q < " << Q << "; ++q; @tile(128, @outer, @inner)) {"     << std::endl
         << "    const CeedScalar* in[" << std::max(args.inputCount(), 1) << "];"       << std::endl
         << "    CeedScalar* out[" << std::max(args.outputCount(), 1) << "];"           << std::endl;

      // Set and define in for the q point
      for (int i = 0; i < args.inputCount(); ++i) {
        const CeedInt fieldSize = args.getQfInput(i).size;
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
      for (int i = 0; i < args.outputCount(); ++i) {
        const CeedInt fieldSize = args.getQfOutput(i).size;
        const std::string qOut_i = "qOut_" + ::occa::toString(i);

        ss << "    CeedScalar " << qOut_i << "[" << fieldSize << "];"                   << std::endl
           << "    out[" << i << "] = " << qOut_i << ";"                                << std::endl;
      }

      ss << "    " << qFunctionName << "(ctx, 1, in, out);"                             << std::endl;

      // Copy out for the q point
      for (int i = 0; i < args.outputCount(); ++i) {
        const CeedInt fieldSize = args.getQfOutput(i).size;
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

    void QFunction::syncContext() {
      if (!ceedContextSize) {
        return;
      }

      if (!qFunctionContext.isInitialized()) {
        qFunctionContext = getDevice().malloc(ceedContextSize);
      }
      qFunctionContext.copyFrom(ceedContext);
    }

    int QFunction::apply(CeedInt Q, CeedVector *U, CeedVector *V) {
      int ierr;
      ierr = buildKernel(Q); CeedChk(ierr);
      syncContext();

      std::vector<CeedScalar*> inputArgs, outputArgs;

      qFunctionKernel.clearArgs();

      for (CeedInt i = 0; i < args.inputCount(); i++) {
        Vector *u = Vector::from(U[i]);
        if (!u) {
          return CeedError(ceed, 1, "Incorrect qFunction input field: U[%i]", (int) i);
        }
        CeedScalar *inputArg;
        ierr = u->getArray(CEED_MEM_DEVICE, &inputArg); CeedChk(ierr);

        inputArgs.push_back(inputArg);
        qFunctionKernel.pushArg(arrayToMemory(inputArg));
      }

      for (CeedInt i = 0; i < args.outputCount(); i++) {
        Vector *v = Vector::from(V[i]);
        if (!v) {
          return CeedError(ceed, 1, "Incorrect qFunction output field: V[%i]", (int) i);
        }
        CeedScalar *outputArg;
        ierr = v->getArray(CEED_MEM_DEVICE, &outputArg); CeedChk(ierr);

        outputArgs.push_back(outputArg);
        qFunctionKernel.pushArg(arrayToMemory(outputArg));
      }

      if (qFunctionContext.isInitialized()) {
        qFunctionKernel.pushArg(qFunctionContext);
      } else {
        qFunctionKernel.pushArg(::occa::null);
      }

      qFunctionKernel.run();

      for (CeedInt i = 0; i < args.inputCount(); i++) {
        Vector *u = Vector::from(U[i]);
        ierr = u->restoreArray(&inputArgs[i]); CeedChk(ierr);
      }

      for (CeedInt i = 0; i < args.outputCount(); i++) {
        Vector *v = Vector::from(V[i]);
        ierr = v->restoreArray(&outputArgs[i]); CeedChk(ierr);
      }

      return 0;
    }

    //---[ Ceed Callbacks ]-----------
    int QFunction::registerQFunctionFunction(Ceed ceed, CeedQFunction qf,
                                             const char *fname, ceed::occa::ceedFunction f) {
      return CeedSetBackendFunction(ceed, "QFunction", qf, fname, f);
    }

    int QFunction::ceedCreate(CeedQFunction qf) {
      int ierr;
      Ceed ceed;
      ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
      Context *context;
      ierr = CeedGetData(ceed, (void**) &context); CeedChk(ierr);
      char *source;
      ierr = CeedQFunctionGetSourcePath(qf, &source); CeedChk(ierr);

      QFunction *qFunction = new QFunction(source);
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
