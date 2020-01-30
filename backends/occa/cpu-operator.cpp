// Copyright (c) 2020, Lawrence Livermore National Security, LLC.
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

#include "cpu-operator.hpp"
#include "cpu-operator-kernel-builder.hpp"
#include "qfunction.hpp"


namespace ceed {
  namespace occa {
    CpuOperator::CpuOperator() {}

    CpuOperator::~CpuOperator() {}

    void CpuOperator::initialSetup() {
      // ierr = CeedCalloc(numinputfields + numoutputfields, &impl->blkrestr);
      // CeedChk(ierr);
      // ierr = CeedCalloc(numinputfields + numoutputfields, &impl->evecs);
      // CeedChk(ierr);
      // ierr = CeedCalloc(numinputfields + numoutputfields, &impl->edata);
      // CeedChk(ierr);

      // // Max 16 inputs
      // ierr = CeedCalloc(16, &impl->inputstate); CeedChk(ierr);
      // ierr = CeedCalloc(16, &impl->evecsin); CeedChk(ierr);
      // ierr = CeedCalloc(16, &impl->evecsout); CeedChk(ierr);
      // ierr = CeedCalloc(16, &impl->qvecsin); CeedChk(ierr);
      // ierr = CeedCalloc(16, &impl->qvecsout); CeedChk(ierr);

      // ierr = CeedOperatorSetupFields_Opt(qf, op, 0, blksize, impl->blkrestr,
      //                                    impl->evecs, impl->evecsin,
      //                                    impl->qvecsin, 0,
      //                                    numinputfields, Q);

      // ierr = CeedOperatorSetupFields_Opt(qf, op, 1, blksize, impl->blkrestr,
      //                                    impl->evecs, impl->evecsout,
      //                                    impl->qvecsout, numinputfields,
      //                                    numoutputfields, Q);
    }

    void CpuOperator::setupApply(Vector &in, Vector &out) {
    }

    void CpuOperator::apply(Vector &in, Vector &out) {
      setupApply(in, out);
      apply(args)
      // Loop through elements
      // Output restriction
      // Restore input arrays
    }

    ::occa::kernel CpuOperator::buildApplyKernel() {
      return CpuOperatorKernelBuilder::build(getDevice(),
                                             qfunction->filename,
                                             qfunction->qFunctionName,
                                             ceedQ,
                                             args);
    }
  }
}
