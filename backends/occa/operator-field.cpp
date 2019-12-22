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
      CeedTransposeMode ceedTransposeMode;
      CeedElemRestriction ceedElemRestriction;
      int ierr = 0;

      ierr = CeedOperatorFieldGetBasis(opField, &ceedBasis);
      CeedOccaValidChk(_isValid, ierr);

      ierr = CeedOperatorFieldGetVector(opField, &ceedVector);
      CeedOccaValidChk(_isValid, ierr);

      ierr = CeedOperatorFieldGetLMode(opField, &ceedTransposeMode);
      CeedOccaValidChk(_isValid, ierr);

      ierr = CeedOperatorFieldGetElemRestriction(opField, &ceedElemRestriction);
      CeedOccaValidChk(_isValid, ierr);

      vec = Vector::from(ceedVector);
      basis = Basis::from(ceedBasis);
      elemRestriction = ElemRestriction::from(ceedElemRestriction);
      isTransposed = (ceedTransposeMode != CEED_NOTRANSPOSE);
    }

    bool OperatorField::isValid() const {
      return _isValid;
    }
  }
}
