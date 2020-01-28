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

#ifndef CEED_OCCA_OPERATORKERNELBUILDER_HEADER
#define CEED_OCCA_OPERATORKERNELBUILDER_HEADER

#include <sstream>

#include "operator-args.hpp"

namespace ceed {
  namespace occa {
    class OperatorKernelBuilder {
     private:
      const std::string qfunctionFilename;
      const std::string qfunctionName;
      const CeedInt Q;
      OperatorArgs args;

      // Helper methods
      std::stringstream ss;
      std::string tab;
      bool varIsInput;
      int varIndex;

     public:
      OperatorKernelBuilder(const std::string &qfunctionFilename_,
                            const std::string &qfunctionName_,
                            const CeedInt Q_,
                            const OperatorArgs &args_);

      ::occa::kernel buildKernel(::occa::device device);

      void operatorKernelArguments();
      void operatorKernelArgument(const int index,
                                  const bool isInput,
                                  const OperatorField &opField,
                                  const QFunctionField &qfField);

      void variableSetup();
      void readQuads();
      void qfunctionArgs();
      void writeQuads();

      //---[ None ]---------------------
      void noneReadQuads(const int index,
                         const OperatorField &opField,
                         const QFunctionField &qfField);

      void noneWriteQuads(const int index,
                          const OperatorField &opField,
                          const QFunctionField &qfField);
      //================================

      //---[ Interp ]-------------------
      void interpReadQuads(const int index,
                           const OperatorField &opField,
                           const QFunctionField &qfField);

      void interpWriteQuads(const int index,
                            const OperatorField &opField,
                            const QFunctionField &qfField);
      //================================

      //---[ Grad ]---------------------
      void gradReadQuads(const int index,
                         const OperatorField &opField,
                         const QFunctionField &qfField);

      void gradWriteQuads(const int index,
                          const OperatorField &opField,
                          const QFunctionField &qfField);
      //================================

      //---[ Weight ]-------------------
      void weightReadQuads(const int index,
                           const OperatorField &opField,
                           const QFunctionField &qfField);

      void weightWriteQuads(const int index,
                            const OperatorField &opField,
                            const QFunctionField &qfField);
      //================================

      //---[ Code ]---------------------
      void indent();
      void unindent();

      void startElementForLoop();
      void endElementForLoop();
      //================================

      //---[ Variables ]----------------
      void setVarInfo(const bool isInput,
                      const int index);

      inline void setInput(const int index) {
        setVarInfo(true, index);
      }

      inline void setOutput(const int index) {
        setVarInfo(false, index);
      }

#define CEED_OCCA_DEFINE_VAR(VAR)                                 \
      inline std::string VAR(const bool isInput,                  \
                             const int index) {                   \
        return var(#VAR, isInput, index);                         \
      }                                                           \
                                                                  \
      inline std::string VAR(const bool isInput,                  \
                             const int index,                     \
                             const std::string &arrayIndex) {     \
        return arrayVar(#VAR, isInput, index, arrayIndex);        \
      }                                                           \
                                                                  \
      inline std::string VAR() {                                  \
        return var(#VAR, varIsInput, varIndex);                   \
      }                                                           \
                                                                  \
      inline std::string VAR(const std::string &arrayIndex) {     \
        return arrayVar(#VAR, varIsInput, varIndex, arrayIndex);  \
      }

      // Arguments
      CEED_OCCA_DEFINE_VAR(B)
      CEED_OCCA_DEFINE_VAR(G)
      CEED_OCCA_DEFINE_VAR(W)
      CEED_OCCA_DEFINE_VAR(field)
      CEED_OCCA_DEFINE_VAR(indices)

      // Field-specific information
      CEED_OCCA_DEFINE_VAR(componentCount)
      CEED_OCCA_DEFINE_VAR(P)
      CEED_OCCA_DEFINE_VAR(s_B)
      CEED_OCCA_DEFINE_VAR(s_G)

      // Assumes Q is the same in all fields
      // CEED_OCCA_DEFINE_VAR(Q)

      // Intermediate values (dofs/quads)
      CEED_OCCA_DEFINE_VAR(r_fieldDofs)      // d_u / d_v
      CEED_OCCA_DEFINE_VAR(r_fieldQuads)     // r_t
      CEED_OCCA_DEFINE_VAR(r_fieldGradQuads) // r_u

      // QFunction arguments
      CEED_OCCA_DEFINE_VAR(r_qfField)        // r_q / r_qq


#undef CEED_OCCA_DEFINE_VAR

      std::string var(const std::string &varName,
                      const bool isInput,
                      const int index);

      std::string arrayVar(const std::string &varName,
                           const bool isInput,
                           const int index,
                           const std::string &arrayIndex);
      //================================


      //---[ Methods ]------------------
      void cacheArray(const std::string &ptr,
                      const std::string &sharedPtr);
      //================================

      static ::occa::kernel build(const ::occa::device &device,
                                  const std::string &qfunctionFilename,
                                  const std::string &qfunctionName,
                                  const CeedInt Q,
                                  const OperatorArgs &args);
    };
  }
}

#endif
