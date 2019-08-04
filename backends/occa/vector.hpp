// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC.
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

#ifndef CEED_OCCA_VECTOR_HEADER
#define CEED_OCCA_VECTOR_HEADER

#include <ceed-backend.h>

namespace ceed {
  namespace occa {
    class Vector {
     public:
      ::occa::memory memory;
      void *hostBuffer;

      Vector(CeedVector vec) {
      }

      void initialize(::occa::device device,
                      size_t bytes) {
        memory = device.malloc(bytes);
      }

      int setArray(CeedMemType mtype,
                   CeedCopyMode cmode, CeedScalar *array) {
        return 0;
      }

      int getArray(CeedMemType mtype,
                   CeedScalar *&array) {
        return 0;
      }

      int getArrayRead(CeedMemType mtype,
                       const CeedScalar *&array) {
        return 0;
      }

      int restoreArray() {
        return 0;
      }

      int restoreArrayRead() {
        return 0;
      }

      int destroy() {
        memory.free();
        return 0;
      }

      //---[ Ceed Callbacks ]-----------
      static int staticSetArray(CeedVector vec, CeedMemType mtype,
                                CeedCopyMode cmode, CeedScalar *array) {
        return Vector(vec).setArray(mtype, cmode, array);
      }

      static int staticGetArray(CeedVector vec, CeedMemType mtype,
                                CeedScalar **array) {
        return Vector(vec).getArray(mtype, *array);
      }

      static int staticGetArrayRead(CeedVector vec, CeedMemType mtype,
                                    const CeedScalar **array) {
        return Vector(vec).getArrayRead(mtype, *array);
      }

      static int staticRestoreArray(CeedVector vec) {
        return Vector(vec).restoreArray();
      }

      static int staticRestoreArrayRead(CeedVector vec) {
        return Vector(vec).restoreArrayRead();
      }

      static int staticDestroy(CeedVector vec) {
        return Vector(vec).destroy();
      }

      //---[ Registration ]-------------
      static int registerVectorFunction(Ceed ceed, CeedVector vec,
                                        const char *fname, ceed::occa::ceedFunction f) {
        return CeedSetBackendFunction(ceed, "Vector", vec, fname, f);
      }

      static int createVector(CeedInt n, CeedVector vec) {
        int ierr;

        Ceed ceed;
        ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);
        Context *context;
        ierr = CeedGetData(ceed, (void**) &context); CeedChk(ierr);
        CeedInt length;
        CeedVectorGetLength(vec, &length);

        ierr = registerVectorFunction(ceed, vec, "SetArray",
                                      (ceed::occa::ceedFunction) Vector::staticSetArray);
        CeedChk(ierr);

        ierr = registerVectorFunction(ceed, vec, "GetArray",
                                      (ceed::occa::ceedFunction) Vector::staticGetArray);
        CeedChk(ierr);

        ierr = registerVectorFunction(ceed, vec, "GetArrayRead",
                                      (ceed::occa::ceedFunction) Vector::staticGetArrayRead);
        CeedChk(ierr);

        ierr = registerVectorFunction(ceed, vec, "RestoreArray",
                                      (ceed::occa::ceedFunction) Vector::staticRestoreArray);
        CeedChk(ierr);

        ierr = registerVectorFunction(ceed, vec, "RestoreArrayRead",
                                      (ceed::occa::ceedFunction) Vector::staticRestoreArrayRead);
        CeedChk(ierr);

        ierr = registerVectorFunction(ceed, vec, "Destroy",
                                      (ceed::occa::ceedFunction) Vector::staticDestroy);
        CeedChk(ierr);

        Vector *vector;
        ierr = CeedCalloc(1, &vector); CeedChk(ierr);

        vector->initialize(context->device,
                           length * sizeof(CeedScalar));

        ierr = CeedVectorSetData(vec, (void**) &vector); CeedChk(ierr);

        return 0;
      }
    };
  }
}

#endif
