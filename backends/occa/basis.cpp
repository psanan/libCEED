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
        ceed(NULL),
        ceedDim(0),
        ceedQuadraturePointCount(0),
        ceedNodeCount(0),
        ceedComponentCount(0) {
      OCCA_DEBUG_TRACE("basis: Basis");
    }

    Basis::~Basis() {
      OCCA_DEBUG_TRACE("basis: ~Basis");
    }

    Basis* Basis::from(CeedBasis basis) {
      OCCA_DEBUG_TRACE("basis: from");
      if (!basis) {
        return NULL;
      }

      int ierr;
      Basis *basis_;

      ierr = CeedBasisGetData(basis, (void**) &basis_); CeedOccaFromChk(ierr);
      ierr = basis_->setCeedFields(basis); CeedOccaFromChk(ierr);

      return basis_;
    }

    Basis* Basis::from(CeedOperatorField operatorField) {
      OCCA_DEBUG_TRACE("basis: from");

      int ierr;
      CeedBasis basis;
      ierr = CeedOperatorFieldGetBasis(operatorField, &basis); CeedOccaFromChk(ierr);
      return from(basis);
    }

    int Basis::setCeedFields(CeedBasis basis) {
      OCCA_DEBUG_TRACE("basis: setCeedFields");

      int ierr;
      ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
      ierr = CeedBasisGetDimension(basis, &ceedDim); CeedChk(ierr);
      ierr = CeedBasisGetNumComponents(basis, &ceedComponentCount); CeedChk(ierr);
      ierr = CeedBasisGetNumNodes(basis, &ceedNodeCount); CeedChk(ierr);

      if (dynamic_cast<TensorBasis*>(this)) {
        ierr = CeedBasisGetNumQuadraturePoints1D(basis, &ceedQuadraturePointCount);
      } else {
        ierr = CeedBasisGetNumQuadraturePoints(basis, &ceedQuadraturePointCount);
      }
      CeedChk(ierr);

      return 0;
    }

    ::occa::device Basis::getDevice() {
      OCCA_DEBUG_TRACE("basis: getDevice");

      return Context::from(ceed)->device;
    }

    //---[ Ceed Callbacks ]-----------
    int Basis::registerBasisFunction(Ceed ceed, CeedBasis basis,
                                     const char *fname, ceed::occa::ceedFunction f) {
      OCCA_DEBUG_TRACE("basis: registerBasisFunction");

      return CeedSetBackendFunction(ceed, "Basis", basis, fname, f);
    }

    int Basis::ceedApply(CeedBasis basis, const CeedInt nelem,
                         CeedTransposeMode tmode,
                         CeedEvalMode emode, CeedVector u, CeedVector v) {
      OCCA_DEBUG_TRACE("basis: ceedApply");

      Basis *basis_ = Basis::from(basis);

      if (!basis_) {
        return CeedError(NULL, 1, "Incorrect CeedBasis argument: op");
      }

      return basis_->apply(
        nelem,
        tmode, emode,
        Vector::from(u), Vector::from(v)
      );
    }

    int Basis::ceedDestroy(CeedBasis basis) {
      OCCA_DEBUG_TRACE("basis: ceedDestroy");

      delete Basis::from(basis);
      return 0;
    }
  }
}
