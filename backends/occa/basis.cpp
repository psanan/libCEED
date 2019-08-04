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

#include "basis.hpp"
#include "tensor-basis.hpp"


namespace ceed {
  namespace occa {
    Basis::Basis() :
        ceedDim(0),
        ceedQuadraturePoints(0),
        ceedNodes(0),
        ceedComponents(0) {}

    Basis::~Basis() {
      interpKernel.free();
      gradKernel.free();
      weightKernel.free();
      interp.free();
      grad.free();
      qWeight.free();
    }

    Basis* Basis::from(CeedBasis basis) {
      int ierr;
      Basis *basis_;

      ierr = CeedBasisGetData(basis, (void**) &basis_); CeedOccaFromChk(ierr);
      ierr = CeedBasisGetCeed(basis, &basis_->ceed); CeedOccaFromChk(ierr);
      ierr = CeedBasisGetDimension(basis, &basis_->ceedDim); CeedOccaFromChk(ierr);
      ierr = CeedBasisGetNumComponents(basis, &basis_->ceedComponents); CeedOccaFromChk(ierr);
      ierr = CeedBasisGetNumNodes(basis, &basis_->ceedNodes); CeedOccaFromChk(ierr);

      if (dynamic_cast<TensorBasis*>(basis_)) {
        ierr = CeedBasisGetNumQuadraturePoints1D(basis, &basis_->ceedQuadraturePoints);
      } else {
        ierr = CeedBasisGetNumQuadraturePoints(basis, &basis_->ceedQuadraturePoints);
      }
      CeedOccaFromChk(ierr);

      return basis_;
    }

    Basis* Basis::from(CeedOperatorField operatorField) {
      int ierr;
      CeedBasis basis;
      ierr = CeedOperatorFieldGetBasis(operatorField, &basis); CeedOccaFromChk(ierr);
      return from(basis);
    }

    ::occa::device Basis::getDevice() {
      return Context::from(ceed).device;
    }

    //---[ Ceed Callbacks ]-----------
    int Basis::registerBasisFunction(Ceed ceed, CeedBasis basis,
                                     const char *fname, ceed::occa::ceedFunction f) {
      return CeedSetBackendFunction(ceed, "Basis", basis, fname, f);
    }

    int Basis::ceedApply(CeedBasis basis, const CeedInt nelem,
                         CeedTransposeMode tmode,
                         CeedEvalMode emode, CeedVector u, CeedVector v) {
      Basis *basis_ = Basis::from(basis);
      Vector *uVector = u ? Vector::from(u) : NULL;
      Vector *vVector = v ? Vector::from(v) : NULL;

      if (!basis_) {
        return CeedError(NULL, 1, "Incorrect CeedBasis argument: op");
      }
      if (u && !uVector) {
        return CeedError(basis_->ceed, 1, "Incorrect CeedVector argument: u");
      }
      if (v && !vVector) {
        return CeedError(basis_->ceed, 1, "Incorrect CeedVector argument: v");
      }

      return basis_->apply(nelem, tmode, emode, uVector, vVector);
    }

    int Basis::ceedDestroy(CeedBasis basis) {
      delete Basis::from(basis);
      return 0;
    }
  }
}
