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

#include "operator-field.hpp"
#include "basis.hpp"
#include "elem-restriction.hpp"
#include "vector.hpp"


namespace ceed {
  namespace occa {
    OperatorField::OperatorField(CeedOperatorField opField) :
        _isValid(false),
        vec(NULL),
        basis(NULL),
        elemRestriction(NULL) {

      CeedBasis ceedBasis;
      CeedVector ceedVector;
      CeedElemRestriction ceedElemRestriction;
      int ierr = 0;

      ierr = CeedOperatorFieldGetBasis(opField, &ceedBasis);
      CeedOccaValidChk(_isValid, ierr);

      ierr = CeedOperatorFieldGetVector(opField, &ceedVector);
      CeedOccaValidChk(_isValid, ierr);

      ierr = CeedOperatorFieldGetElemRestriction(opField, &ceedElemRestriction);
      CeedOccaValidChk(_isValid, ierr);

      _isValid = true;

      vec = Vector::from(ceedVector);
      basis = Basis::from(ceedBasis);
      elemRestriction = ElemRestriction::from(ceedElemRestriction);
    }

    bool OperatorField::isValid() const {
      return _isValid;
    }

    //---[ Basis Info ]-----------------
    int OperatorField::getP() const {
      return (
        basis
        ? basis->ceedNodeCount
        : 0
      );
    }

    int OperatorField::getQ() const {
      return (
        basis
        ? basis->ceedQuadraturePointCount
        : 0
      );
    }

    int OperatorField::getComponentCount() const {
      return (
        basis
        ? basis->ceedComponentCount
        : 1
      );
    }
    //==================================

    //---[ ElemRestriction Info ]-------
    int OperatorField::getElementCount() const {
      return (
        elemRestriction
        ? elemRestriction->ceedElementCount
        : 1
      );
    }

    int OperatorField::getElementSize() const {
      return (
        elemRestriction
        ? elemRestriction->ceedElementSize
        : 1
      );
    }

    int OperatorField::getNodeCount() const {
      return (
        elemRestriction
        ? elemRestriction->ceedNodeCount
        : 1
      );
    }
    //==================================
  }
}
