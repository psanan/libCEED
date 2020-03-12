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

#ifndef CEED_OCCA_GPUOPERATORKERNELBUILDER_HEADER
#define CEED_OCCA_GPUOPERATORKERNELBUILDER_HEADER

#include "operator-kernel-builder.hpp"

namespace ceed {
  namespace occa {
    class GpuOperatorKernelBuilder : public OperatorKernelBuilder {
     public:
      GpuOperatorKernelBuilder(const std::string &qfunctionFilename_,
                               const std::string &qfunctionName_,
                               const CeedInt Q_,
                               const OperatorArgs &args_);

      ::occa::properties getKernelProps(::occa::device device);

      void generateKernel(::occa::device device);

      void operatorKernelArguments();
      void operatorKernelArgument(const int index,
                                  const bool isInput,
                                  const OperatorField &opField,
                                  const QFunctionField &qfField);

      static ::occa::kernel build(const ::occa::device &device,
                                  const std::string &qfunctionFilename,
                                  const std::string &qfunctionName,
                                  const CeedInt Q,
                                  const OperatorArgs &args);
    };
  }
}

#endif
