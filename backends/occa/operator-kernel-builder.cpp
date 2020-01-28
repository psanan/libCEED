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
        qfunctionFilename(qfunctionFilename_),
        qfunctionName(qfunctionName_),
        Q(Q_),
        args(args_) {}

    ::occa::kernel OperatorKernelBuilder::buildKernel(::occa::device device) {
      // B = basis_data->d_interp1d;
      // G = basis_data->d_collograd1d;
      // G = basis_data->d_grad1d;
      // W = basis_data->d_qweight1d;
      // indices = restr_data->d_ind;

      const std::string kernelName = "operator";
      ss.str("");

      ss << tab << "@kernel"                    << std::endl
         << tab << "void " << kernelName << "(" << std::endl;
      operatorKernelArguments();
      ss << tab << ") {"                        << std::endl;
      indent();

      ss << tab << "for (int eBlock = 0; eBlock < elementCount; eBlock += ELEMENTS_PER_BLOCK; @outer) {" << std::endl;
      indent();

      ss << tab << "@tile(Q, @inner, @inner)"                  << std::endl
         << tab << "for (int tid = 0; tid < (Q * Q); ++tid) {" << std::endl;
      indent();

      // Kernel body
      // TODO

      unindent();
      ss << tab << "}" << std::endl; // @inner
      unindent();
      ss << tab << "}" << std::endl; // @outer
      unindent();
      ss << tab << "}" << std::endl; // @kernel

      const std::string source = ss.str();

      std::cout << "source:\n\n" << source << "\n\n";
      throw 1;

      return device.buildKernelFromString(source,
                                          kernelName,
                                          QFunction::getKernelProps(qfunctionFilename, Q));
    }

    void OperatorKernelBuilder::operatorKernelArguments() {
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

    void OperatorKernelBuilder::operatorKernelArgument(const int index,
                                                       const bool isInput,
                                                       const OperatorField &opField,
                                                       const QFunctionField &qfField) {
      setVarInfo(isInput, index);

      const std::string qualifiers = isInput ? "" : "const ";
      const std::string scalarPtr = qualifiers + "CeedScalar *";
      const std::string intPtr = qualifiers + "CeedInt *";

      if (isInput) {
        ss << tab << "// Input " << index       << std::endl;
      } else {
        ss << tab << "// Output " << index      << std::endl;
      }

      ss << tab << scalarPtr << field() << ","  << std::endl;

      if (qfField.usesB()) {
        ss << tab << scalarPtr << B() << ","    << std::endl;
      }

      if (qfField.usesG()) {
        ss << tab << scalarPtr << G() << ","    << std::endl;
      }

      if (qfField.usesW()) {
        ss << tab << scalarPtr << W() << ","    << std::endl;
      }

      if (qfField.usesIndices()) {
        ss << tab << intPtr << indices() << "," << std::endl;
      }

      ss << std::endl;
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

    ::occa::kernel OperatorKernelBuilder::build(const ::occa::device &device,
                                                const std::string &qfunctionFilename,
                                                const std::string &qfunctionName,
                                                const CeedInt Q,
                                                const OperatorArgs &args) {
      OperatorKernelBuilder builder(qfunctionFilename, qfunctionName, Q, args);
      return builder.buildKernel(device);
    }
  }
}
