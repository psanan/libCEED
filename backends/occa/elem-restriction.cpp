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

#include "elem-restriction.hpp"


namespace ceed {
  namespace occa {
    ElemRestriction::ElemRestriction() {}

    ElemRestriction::~ElemRestriction() {}

    ElemRestriction* ElemRestriction::from(CeedElemRestriction r) {
      int ierr;
      ElemRestriction *elemRestriction;

      ierr = CeedElemRestrictionGetData(r, (void**) &elemRestriction); CeedOccaFromChk(ierr);
      ierr = CeedElemRestrictionGetCeed(r, &elemRestriction->ceed); CeedOccaFromChk(ierr);

      ierr = CeedElemRestrictionGetNumElements(r, &elemRestriction->ceedElementCount);
      CeedOccaFromChk(ierr);

      ierr = CeedElemRestrictionGetElementSize(r, &elemRestriction->ceedElementSize);
      CeedOccaFromChk(ierr);

      ierr = CeedElemRestrictionGetNumComponents(r, &elemRestriction->ceedComponentCount);
      CeedOccaFromChk(ierr);

      ierr = CeedElemRestrictionGetNumDoF(r, &elemRestriction->ceedDofCount);
      CeedOccaFromChk(ierr);

      ierr = CeedElemRestrictionGetBlockSize(r, &elemRestriction->ceedBlockSize);
      CeedOccaFromChk(ierr);

      return elemRestriction;
    }

    ::occa::device ElemRestriction::getDevice() {
      return Context::from(ceed).device;
    }

    int ElemRestriction::apply(CeedTransposeMode tmode, CeedTransposeMode lmode,
                               Vector &u, Vector &v, CeedRequest *request) {
      // TODO: Implement
      return 0;
    }

    int ElemRestriction::applyBlock(CeedInt block,
                                    CeedTransposeMode tmode, CeedTransposeMode lmode,
                                    Vector &u, Vector &v, CeedRequest *request) {
      // TODO: Implement
      return 0;
    }

    //---[ Ceed Callbacks ]-----------
    int ElemRestriction::registerRestrictionFunction(Ceed ceed, CeedElemRestriction r,
                                                     const char *fname, ceed::occa::ceedFunction f) {
      return CeedSetBackendFunction(ceed, "ElemRestriction", r, fname, f);
    }

    int ElemRestriction::ceedCreate(CeedElemRestriction r) {
      int ierr;

      Ceed ceed;
      ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);

      ElemRestriction *elemRestriction = new ElemRestriction();
      ierr = CeedElemRestrictionSetData(r, (void**) &elemRestriction); CeedChk(ierr);

      ierr = registerRestrictionFunction(ceed, r, "Apply",
                                         (ceed::occa::ceedFunction) ElemRestriction::ceedApply);
      CeedChk(ierr);

      ierr = registerRestrictionFunction(ceed, r, "ApplyBlock",
                                         (ceed::occa::ceedFunction) ElemRestriction::ceedApplyBlock);
      CeedChk(ierr);

      ierr = registerRestrictionFunction(ceed, r, "Destroy",
                                         (ceed::occa::ceedFunction) ElemRestriction::ceedDestroy);
      CeedChk(ierr);

      return 0;
    }

    int ElemRestriction::ceedApply(CeedElemRestriction r,
                                   CeedTransposeMode tmode, CeedTransposeMode lmode,
                                   CeedVector u, CeedVector v, CeedRequest *request) {
      ElemRestriction *elemRestriction = ElemRestriction::from(r);
      Vector *uVector = Vector::from(u);
      Vector *vVector = Vector::from(v);

      if (!elemRestriction) {
        return CeedError(NULL, 1, "Incorrect CeedElemRestriction argument: r");
      }
      if (!uVector) {
        return CeedError(elemRestriction->ceed, 1, "Incorrect CeedVector argument: u");
      }
      if (!vVector) {
        return CeedError(elemRestriction->ceed, 1, "Incorrect CeedVector argument: v");
      }

      return elemRestriction->apply(tmode, lmode, *uVector, *vVector, request);
    }

    int ElemRestriction::ceedApplyBlock(CeedElemRestriction r,
                                        CeedInt block,
                                        CeedTransposeMode tmode, CeedTransposeMode lmode,
                                        CeedVector u, CeedVector v, CeedRequest *request) {
      ElemRestriction *elemRestriction = ElemRestriction::from(r);
      Vector *uVector = Vector::from(u);
      Vector *vVector = Vector::from(v);

      if (!elemRestriction) {
        return CeedError(NULL, 1, "Incorrect CeedElemRestriction argument: r");
      }
      if (!uVector) {
        return CeedError(elemRestriction->ceed, 1, "Incorrect CeedVector argument: u");
      }
      if (!vVector) {
        return CeedError(elemRestriction->ceed, 1, "Incorrect CeedVector argument: v");
      }

      return elemRestriction->applyBlock(block, tmode, lmode, *uVector, *vVector, request);
    }

    int ElemRestriction::ceedDestroy(CeedElemRestriction r) {
      delete ElemRestriction::from(r);
      return 0;
    }
  }
}
