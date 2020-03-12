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

#include "cpu-operator-kernel-builder.hpp"
#include "qfunction.hpp"


// d_u<i> = getFieldVariable(true, i)

namespace ceed {
  namespace occa {
    CpuOperatorKernelBuilder::CpuOperatorKernelBuilder(const std::string &qfunctionFilename_,
                                                       const std::string &qfunctionName_,
                                                       const CeedInt Q_,
                                                       const OperatorArgs &args_) :
        OperatorKernelBuilder(qfunctionFilename_, qfunctionName_, Q_, args_) {}

    ::occa::properties CpuOperatorKernelBuilder::getKernelProps(::occa::device device) {
      return ::occa::properties();
    }

    void CpuOperatorKernelBuilder::generateKernel(::occa::device device) {
      ss << tab << "@kernel"                    << std::endl
         << tab << "void " << kernelName << "(" << std::endl;
      operatorKernelArguments();
      ss << tab << ") {"                        << std::endl;
      indent();

      ss << tab << "for (int e = 0; e < elementCount; ++e; @outer) {" << std::endl;
      indent();

      // Kernel body
      // TODO

      unindent();
      ss << tab << "}" << std::endl; // @outer
      unindent();
      ss << tab << "}" << std::endl; // @kernel
    }

    void CpuOperatorKernelBuilder::operatorKernelArguments() {
      for (int i = 0; i < args.inputCount(); ++i) {
        operatorKernelArgument(i, true, args.getOpInput(i), args.getQfInput(i));
      }

      for (int i = 0; i < args.outputCount(); ++i) {
        operatorKernelArgument(i, false, args.getOpOutput(i), args.getQfOutput(i));
      }

      ss << "  // Extra params"             << std::endl
         << "  const CeedInt elementCount," << std::endl
         << "  const void *ctx"             << std::endl;
    }

    void CpuOperatorKernelBuilder::operatorKernelArgument(const int index,
                                                          const bool isInput,
                                                          const OperatorField &opField,
                                                          const QFunctionField &qfField) {
      // setVarInfo(isInput, index);

      // const std::string qualifiers = isInput ? "" : "const ";
      // const std::string scalarPtr = qualifiers + "CeedScalar *";
      // const std::string intPtr = qualifiers + "CeedInt *";

      // if (isInput) {
      //   ss << tab << "// Input " << index       << std::endl;
      // } else {
      //   ss << tab << "// Output " << index      << std::endl;
      // }

      // ss << tab << scalarPtr << field() << ","  << std::endl;

      // if (qfField.usesB()) {
      //   ss << tab << scalarPtr << B() << ","    << std::endl;
      // }

      // if (qfField.usesG()) {
      //   ss << tab << scalarPtr << G() << ","    << std::endl;
      // }

      // if (qfField.usesW()) {
      //   ss << tab << scalarPtr << W() << ","    << std::endl;
      // }

      // if (qfField.usesIndices()) {
      //   ss << tab << intPtr << indices() << "," << std::endl;
      // }

      // ss << std::endl;
    }

    ::occa::kernel CpuOperatorKernelBuilder::build(const ::occa::device &device,
                                                   const std::string &qfunctionFilename,
                                                   const std::string &qfunctionName,
                                                   const CeedInt Q,
                                                   const OperatorArgs &args) {
      CpuOperatorKernelBuilder builder(qfunctionFilename, qfunctionName, Q, args);
      return builder.buildKernel(device);
    }
  }
}
