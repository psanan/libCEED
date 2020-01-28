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

#include "gpu-operator-kernel-builder.hpp"
#include "qfunction.hpp"


// d_u<i> = getFieldVariable(true, i)

namespace ceed {
  namespace occa {
    GpuOperatorKernelBuilder::GpuOperatorKernelBuilder(const std::string &qfunctionFilename_,
                                                       const std::string &qfunctionName_,
                                                       const CeedInt Q_,
                                                       const OperatorArgs &args_) :
        OperatorKernelBuilder(qfunctionFilename_, qfunctionName_, Q_, args_) {}

    ::occa::properties GpuOperatorKernelBuilder::getKernelProps(::occa::device device) {
      return QFunction::getKernelProps(qfunctionFilename, Q);
    }

    void GpuOperatorKernelBuilder::generateKernel(::occa::device device) {
      // TODO;
    }

    ::occa::kernel GpuOperatorKernelBuilder::build(const ::occa::device &device,
                                                   const std::string &qfunctionFilename,
                                                   const std::string &qfunctionName,
                                                   const CeedInt Q,
                                                   const OperatorArgs &args) {
      GpuOperatorKernelBuilder builder(qfunctionFilename, qfunctionName, Q, args);
      return builder.buildKernel(device);
    }
  }
}
