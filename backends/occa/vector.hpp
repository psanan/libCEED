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

#include "types.hpp"


namespace ceed {
  namespace occa {
    typedef enum {
      HOST_SYNC,
      DEVICE_SYNC,
      BOTH_SYNC,
      NONE_SYNC
    } VectorSyncState;

    class Vector {
     public:
      // Ceed object information
      Ceed ceed;
      CeedInt ceedLength;

      // Owned resource
      ::occa::memory memory;
      CeedInt hostBufferLength;
      CeedScalar *hostBuffer;

      // Current resources
      ::occa::memory currentMemory;
      CeedScalar *currentHostBuffer;

      // State information
      VectorSyncState syncState;

      Vector() :
          hostBufferLength(0),
          hostBuffer(NULL),
          currentHostBuffer(NULL),
          syncState(NONE_SYNC) {}

      ~Vector() {
        memory.free();
        freeHostBuffer();
      }

      static Vector* from(CeedVector vec) {
        int ierr;
        Vector *vector;

#define CeedVectorChk(IERR) if (IERR) return NULL
        ierr = CeedVectorGetData(vec, (void**) &vector); CeedVectorChk(ierr);
        ierr = CeedVectorGetCeed(vec, &vector->ceed); CeedVectorChk(ierr);
        ierr = CeedVectorGetLength(vec, &vector->ceedLength); CeedVectorChk(ierr);
#undef CeedVectorChk

        return vector;
      }

      ::occa::device getDevice() {
        if (memory.isInitialized()) {
          return memory.getDevice();
        }

        Context *context;
        CeedGetData(ceed, (void**) &context);
        return context->device;
      }

      void resizeMemory(const CeedInt length) {
        resizeMemory(getDevice(), length);
      }

      void resizeMemory(::occa::device device, const CeedInt length) {
        if (length != (CeedInt) memory.length()) {
          memory.free();
          memory = device.malloc(ceedLength, ::occa::dtype::get<CeedScalar>());
        }
      }

      void resizeHostBuffer(const CeedInt length) {
        if (length != hostBufferLength) {
          delete hostBuffer;
          hostBuffer = new CeedScalar[length];
        }
      }

      void setCurrentMemoryIfNeeded() {
        if (!currentMemory.isInitialized()) {
          resizeMemory(ceedLength);
          currentMemory = memory;
        }
      }

      void setCurrentHostBufferIfNeeded() {
        if (!currentHostBuffer) {
          resizeHostBuffer(ceedLength);
          currentHostBuffer = hostBuffer;
        }
      }

      void freeHostBuffer() {
        if (hostBuffer) {
          delete [] hostBuffer;
          hostBuffer = NULL;
        }
      }

      int setArray(CeedMemType mtype,
                   CeedCopyMode cmode, CeedScalar *array) {
        switch (cmode) {
          case CEED_COPY_VALUES:
            return copyArrayValues(mtype, array);
          case CEED_OWN_POINTER:
            return ownArrayPointer(mtype, array);
          case CEED_USE_POINTER:
            return useArrayPointer(mtype, array);
        }
        return 1;
      }

      int copyArrayValues(CeedMemType mtype, CeedScalar *array) {
        switch (mtype) {
          case CEED_MEM_HOST:
            setCurrentHostBufferIfNeeded();
            if (array) {
              memcpy(currentHostBuffer, array, ceedLength * sizeof(CeedScalar));
            }
            return 0;
          case CEED_MEM_DEVICE:
            setCurrentMemoryIfNeeded();
            if (array) {
              currentMemory.copyFrom(array);
            }
            return 0;
        }
        return 1;
      }

      int ownArrayPointer(CeedMemType mtype, CeedScalar *array) {
        switch (mtype) {
          case CEED_MEM_HOST:
            freeHostBuffer();
            hostBuffer = currentHostBuffer = array;
            return 0;
          case CEED_MEM_DEVICE:
            memory.free();
            memory = currentMemory = (::occa::modeMemory_t*) array;
            return 0;
        }
        return 1;
      }

      int useArrayPointer(CeedMemType mtype, CeedScalar *array) {
        switch (mtype) {
          case CEED_MEM_HOST:
            freeHostBuffer();
            currentHostBuffer = array;
            return 0;
          case CEED_MEM_DEVICE:
            memory.free();
            currentMemory = (::occa::modeMemory_t*) array;
            return 0;
        }
        return 1;
      }

      int getArray(CeedMemType mtype,
                   const CeedScalar **array) {
        switch (mtype) {
          case CEED_MEM_HOST:
            setCurrentHostBufferIfNeeded();
            if (syncState == DEVICE_SYNC) {
              setCurrentMemoryIfNeeded();
              currentMemory.copyTo(currentHostBuffer);
              syncState = HOST_SYNC;
            }
            *array = currentHostBuffer;
            return 0;
          case CEED_MEM_DEVICE:
            setCurrentMemoryIfNeeded();
            if (syncState == HOST_SYNC) {
              setCurrentHostBufferIfNeeded();
              currentMemory.copyFrom(currentHostBuffer);
              syncState = DEVICE_SYNC;
            }
            *array = (CeedScalar*) currentMemory.getModeMemory();
            return 0;
        }
        return 1;
      }

      int restoreArray() {
        return 0;
      }

      int restoreArrayRead() {
        return 0;
      }

      //---[ Ceed Callbacks ]-----------
      static int ceedSetArray(CeedVector vec, CeedMemType mtype,
                              CeedCopyMode cmode, CeedScalar *array) {
        Vector *vector = Vector::from(vec);
        if (vector) {
          return vector->setArray(mtype, cmode, array);
        }
        return 1;
      }

      static int ceedGetArray(CeedVector vec, CeedMemType mtype,
                              CeedScalar **array) {
        Vector *vector = Vector::from(vec);
        if (vector) {
          return vector->getArray(mtype, (const CeedScalar**) array);
        }
        return 1;
      }

      static int ceedGetArrayRead(CeedVector vec, CeedMemType mtype,
                                  const CeedScalar **array) {
        Vector *vector = Vector::from(vec);
        if (vector) {
          return vector->getArray(mtype, array);
        }
        return 1;
      }

      static int ceedRestoreArray(CeedVector vec) {
        Vector *vector = Vector::from(vec);
        if (vector) {
          return vector->restoreArray();
        }
        return 1;
      }

      static int ceedRestoreArrayRead(CeedVector vec) {
        Vector *vector = Vector::from(vec);
        if (vector) {
          return vector->restoreArrayRead();
        }
        return 1;
      }

      static int ceedDestroy(CeedVector vec) {
        delete Vector::from(vec);
        return 0;
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

        ierr = registerVectorFunction(ceed, vec, "SetArray",
                                      (ceed::occa::ceedFunction) Vector::ceedSetArray);
        CeedChk(ierr);

        ierr = registerVectorFunction(ceed, vec, "GetArray",
                                      (ceed::occa::ceedFunction) Vector::ceedGetArray);
        CeedChk(ierr);

        ierr = registerVectorFunction(ceed, vec, "GetArrayRead",
                                      (ceed::occa::ceedFunction) Vector::ceedGetArrayRead);
        CeedChk(ierr);

        ierr = registerVectorFunction(ceed, vec, "RestoreArray",
                                      (ceed::occa::ceedFunction) Vector::ceedRestoreArray);
        CeedChk(ierr);

        ierr = registerVectorFunction(ceed, vec, "RestoreArrayRead",
                                      (ceed::occa::ceedFunction) Vector::ceedRestoreArrayRead);
        CeedChk(ierr);

        ierr = registerVectorFunction(ceed, vec, "Destroy",
                                      (ceed::occa::ceedFunction) Vector::ceedDestroy);
        CeedChk(ierr);

        Vector *vector = new Vector();
        ierr = CeedVectorSetData(vec, (void**) &vector); CeedChk(ierr);

        return 0;
      }
    };
  }
}

#endif
