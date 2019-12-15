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

#include "vector.hpp"


namespace ceed {
  namespace occa {
    ::occa::memory arrayToMemory(CeedScalar *array) {
      if (array) {
        return ::occa::memory((::occa::modeMemory_t*) array);
      }
      return ::occa::memory();
    }

    CeedScalar* memoryToArray(::occa::memory &memory) {
      return (CeedScalar*) memory.getModeMemory();
    }

    Vector::Vector() :
        ceed(NULL),
        length(0),
        hostBufferLength(0),
        hostBuffer(NULL),
        currentHostBuffer(NULL),
        syncState(NONE_SYNC) {}

    Vector::~Vector() {
      memory.free();
      freeHostBuffer();
    }

    Vector* Vector::from(CeedVector vec) {
      if (!vec) {
        return NULL;
      }

      int ierr;

      Vector *vector = NULL;
      ierr = CeedVectorGetData(vec, (void**) &vector); CeedOccaFromChk(ierr);

      if (vector != NULL) {
        ierr = CeedVectorGetCeed(vec, &vector->ceed); CeedOccaFromChk(ierr);
        ierr = CeedVectorGetLength(vec, &vector->length); CeedOccaFromChk(ierr);
      }

      return vector;
    }

    ::occa::device Vector::getDevice() {
      if (memory.isInitialized()) {
        return memory.getDevice();
      }
      return Context::from(ceed)->device;
    }

    void Vector::resize(const CeedInt length_) {
      length = length_;
    }

    void Vector::resizeMemory(const CeedInt length_) {
      resizeMemory(getDevice(), length_);
    }

    void Vector::resizeMemory(::occa::device device, const CeedInt length_) {
      if (length_ != (CeedInt) memory.length()) {
        memory.free();
        memory = device.malloc<CeedScalar>(length_);
      }
    }

    void Vector::resizeHostBuffer(const CeedInt length_) {
      if (length_ != hostBufferLength) {
        delete hostBuffer;
        hostBuffer = new CeedScalar[length_];
      }
    }

    void Vector::setCurrentMemoryIfNeeded() {
      if (!currentMemory.isInitialized()) {
        resizeMemory(length);
        currentMemory = memory;
      }
    }

    void Vector::setCurrentHostBufferIfNeeded() {
      if (!currentHostBuffer) {
        resizeHostBuffer(length);
        currentHostBuffer = hostBuffer;
      }
    }

    void Vector::freeHostBuffer() {
      if (hostBuffer) {
        delete [] hostBuffer;
        hostBuffer = NULL;
      }
    }

    int Vector::setValue(CeedScalar value) {
      if (syncState == HOST_SYNC) {
        setCurrentHostBufferIfNeeded();
        for (CeedInt i = 0; i < length; ++i) {
          currentHostBuffer[i] = value;
        }
      } else {
        setCurrentMemoryIfNeeded();
        ::occa::linalg::operator_eq(currentMemory, value);
        syncState = DEVICE_SYNC;
      }
      return 0;
    }

    int Vector::setArray(CeedMemType mtype,
                         CeedCopyMode cmode, CeedScalar *array) {
      switch (cmode) {
        case CEED_COPY_VALUES:
          return copyArrayValues(mtype, array);
        case CEED_OWN_POINTER:
          return ownArrayPointer(mtype, array);
        case CEED_USE_POINTER:
          return useArrayPointer(mtype, array);
      }
      return CeedError(ceed, 1, "Invalid CeedCopyMode passed");
    }

    int Vector::copyArrayValues(CeedMemType mtype, CeedScalar *array) {
      switch (mtype) {
        case CEED_MEM_HOST:
          setCurrentHostBufferIfNeeded();
          if (array) {
            ::memcpy(currentHostBuffer, array, length * sizeof(CeedScalar));
          }
          syncState = HOST_SYNC;
          return 0;
        case CEED_MEM_DEVICE:
          setCurrentMemoryIfNeeded();
          if (array) {
            currentMemory.copyFrom(arrayToMemory(array));
          }
          syncState = DEVICE_SYNC;
          return 0;
      }
      return CeedError(ceed, 1, "Invalid CeedMemType passed");
    }

    int Vector::ownArrayPointer(CeedMemType mtype, CeedScalar *array) {
      switch (mtype) {
        case CEED_MEM_HOST:
          freeHostBuffer();
          hostBuffer = currentHostBuffer = array;
          syncState = HOST_SYNC;
          return 0;
        case CEED_MEM_DEVICE:
          memory.free();
          memory = currentMemory = arrayToMemory(array);
          syncState = DEVICE_SYNC;
          return 0;
      }
      return CeedError(ceed, 1, "Invalid CeedMemType passed");
    }

    int Vector::useArrayPointer(CeedMemType mtype, CeedScalar *array) {
      switch (mtype) {
        case CEED_MEM_HOST:
          freeHostBuffer();
          currentHostBuffer = array;
          syncState = HOST_SYNC;
          return 0;
        case CEED_MEM_DEVICE:
          memory.free();
          currentMemory = arrayToMemory(array);
          syncState = DEVICE_SYNC;
          return 0;
      }
      return CeedError(ceed, 1, "Invalid CeedMemType passed");
    }

    int Vector::getArray(CeedMemType mtype,
                         CeedScalar **array) {
      return getArray(mtype, (const CeedScalar**) array);
    }

    int Vector::getArray(CeedMemType mtype,
                         const CeedScalar **array) {
      switch (mtype) {
        case CEED_MEM_HOST:
          setCurrentHostBufferIfNeeded();
          if (syncState == DEVICE_SYNC) {
            setCurrentMemoryIfNeeded();
            currentMemory.copyTo(currentHostBuffer);
            syncState = BOTH_SYNC;
          }
          *array = currentHostBuffer;
          return 0;
        case CEED_MEM_DEVICE:
          setCurrentMemoryIfNeeded();
          if (syncState == HOST_SYNC) {
            setCurrentHostBufferIfNeeded();
            currentMemory.copyFrom(currentHostBuffer);
            syncState = BOTH_SYNC;
          }
          *array = memoryToArray(currentMemory);
          return 0;
      }
      return CeedError(ceed, 1, "Invalid CeedMemType passed");
    }

    int Vector::restoreArray(CeedScalar **array) {
      return restoreArray((const CeedScalar**) array);
    }

    int Vector::restoreArray(const CeedScalar **array) {
      return 0;
    }

    ::occa::memory Vector::getKernelArg() {
      setCurrentMemoryIfNeeded();
      if (syncState == HOST_SYNC) {
        setCurrentHostBufferIfNeeded();
        currentMemory.copyFrom(currentHostBuffer);
      }
      syncState = DEVICE_SYNC;
      return currentMemory;
    }

    ::occa::memory Vector::getConstKernelArg() {
      setCurrentMemoryIfNeeded();
      if (syncState == HOST_SYNC) {
        setCurrentHostBufferIfNeeded();
        currentMemory.copyFrom(currentHostBuffer);
        syncState = BOTH_SYNC;
      }
      return currentMemory;
    }

    //---[ Ceed Callbacks ]-----------
    int Vector::registerVectorFunction(Ceed ceed, CeedVector vec,
                                       const char *fname, ceed::occa::ceedFunction f) {
      return CeedSetBackendFunction(ceed, "Vector", vec, fname, f);
    }

    int Vector::ceedCreate(CeedInt length, CeedVector vec) {
      int ierr;

      Ceed ceed;
      ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);

      ierr = registerVectorFunction(ceed, vec, "SetValue",
                                    (ceed::occa::ceedFunction) Vector::ceedSetValue);
      CeedChk(ierr);

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

    int Vector::ceedSetValue(CeedVector vec, CeedScalar value) {
      Vector *vector = Vector::from(vec);
      if (!vector) {
        return CeedError(NULL, 1, "Invalid CeedVector passed");
      }
      return vector->setValue(value);
    }

    int Vector::ceedSetArray(CeedVector vec, CeedMemType mtype,
                             CeedCopyMode cmode, CeedScalar *array) {
      Vector *vector = Vector::from(vec);
      if (!vector) {
        return CeedError(NULL, 1, "Invalid CeedVector passed");
      }
      return vector->setArray(mtype, cmode, array);
    }

    int Vector::ceedGetArray(CeedVector vec, CeedMemType mtype,
                             CeedScalar **array) {
      Vector *vector = Vector::from(vec);
      if (!vector) {
        return CeedError(NULL, 1, "Invalid CeedVector passed");
      }
      return vector->getArray(mtype, array);
    }

    int Vector::ceedGetArrayRead(CeedVector vec, CeedMemType mtype,
                                 const CeedScalar **array) {
      Vector *vector = Vector::from(vec);
      if (!vector) {
        return CeedError(NULL, 1, "Invalid CeedVector passed");
      }
      return vector->getArray(mtype, array);
    }

    int Vector::ceedRestoreArray(CeedVector vec, CeedScalar **array) {
      Vector *vector = Vector::from(vec);
      if (!vector) {
        return CeedError(NULL, 1, "Invalid CeedVector passed");
      }
      return vector->restoreArray(array);
    }

    int Vector::ceedRestoreArrayRead(CeedVector vec, CeedScalar **array) {
      Vector *vector = Vector::from(vec);
      if (!vector) {
        return CeedError(NULL, 1, "Invalid CeedVector passed");
      }
      return vector->restoreArray(array);
    }

    int Vector::ceedDestroy(CeedVector vec) {
      delete Vector::from(vec);
      return 0;
    }
  }
}
