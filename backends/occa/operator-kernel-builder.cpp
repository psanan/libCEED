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

#include "operator-kernel-builder.hpp"
#include "qfunction.hpp"


// d_u<i> = getFieldVariable(true, i)

namespace ceed {
  namespace occa {
    OperatorKernelBuilder::OperatorKernelBuilder(const std::string &qfunctionFilename_,
                                                 const std::string &qfunctionName_,
                                                 const CeedInt Q_,
                                                 const OperatorArgs &args_) :
        kernelName("operator"),
        qfunctionFilename(qfunctionFilename_),
        qfunctionName(qfunctionName_),
        Q(Q_),
        args(args_) {}

    ::occa::kernel OperatorKernelBuilder::buildKernel(::occa::device device) {
      ss.str("");

      generateKernel(device);

      const std::string source = ss.str();

      std::cout << "source:\n\n" << source << "\n\n";
      throw 1;

      return device.buildKernelFromString(source,
                                          kernelName,
                                          getKernelProps(device));
    }

    //---[ Code ]-----------------------
    void OperatorKernelBuilder::indent() {
      tab += "  ";
    }

    void OperatorKernelBuilder::unindent() {
      const int chars = (int) tab.size();
      if (chars > 0) {
        tab.resize(std::min(2, chars));
      }
    }
    //==================================

    //---[ Variables ]------------------
    void OperatorKernelBuilder::setVarInfo(const bool isInput,
                                           const int index) {
      varIsInput = isInput;
      varIndex = index;
    }

    std::string OperatorKernelBuilder::var(const std::string &varName,
                                           const bool isInput,
                                           const int index) {
      std::stringstream ss2;
      ss2 << varName << '_' << (isInput ? "in" : "out") << '_' << index;
      return ss2.str();
    }

    std::string OperatorKernelBuilder::arrayVar(const std::string &varName,
                                                const bool isInput,
                                                const int index,
                                                const std::string &arrayIndex) {
      std::stringstream ss2;
      ss2 << var(varName, isInput, index) << '[' << arrayIndex << ']';
      return ss2.str();
    }
    //==================================
  }
}
