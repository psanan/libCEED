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

#ifndef CEED_OCCA_VECTOR_HEADER
#define CEED_OCCA_VECTOR_HEADER

#include "types.hpp"


namespace ceed {
  namespace occa {
    class Vector {
     public:
      // Ceed object information
      Ceed ceed;

      // Owned resources
      CeedInt length;
      ::occa::memory memory;
      CeedInt hostBufferLength;
      CeedScalar *hostBuffer;

      // Current resources
      ::occa::memory currentMemory;
      CeedScalar *currentHostBuffer;

      // State information
      VectorSyncState syncState;

      Vector();

      ~Vector();

      static Vector* from(CeedVector vec);

      ::occa::device getDevice();

      void resize(const CeedInt length_);

      void resizeMemory(const CeedInt length_);

      void resizeMemory(::occa::device device, const CeedInt length_);

      void resizeHostBuffer(const CeedInt length_);

      void setCurrentMemoryIfNeeded();

      void setCurrentHostBufferIfNeeded();

      void freeHostBuffer();

      int setArray(CeedMemType mtype,
                   CeedCopyMode cmode, CeedScalar *array);

      int copyArrayValues(CeedMemType mtype, CeedScalar *array);

      int ownArrayPointer(CeedMemType mtype, CeedScalar *array);

      int useArrayPointer(CeedMemType mtype, CeedScalar *array);

      int getArray(CeedMemType mtype,
                   CeedScalar **array);

      int getArray(CeedMemType mtype,
                   const CeedScalar **array);

      int restoreArray(CeedScalar **array);

      int restoreArray(const CeedScalar **array);

      operator ::occa::kernelArg();

      //---[ Ceed Callbacks ]-----------
      static int registerVectorFunction(Ceed ceed, CeedVector vec,
                                        const char *fname, ceed::occa::ceedFunction f);

      static int ceedCreate(CeedInt length, CeedVector vec);

      static int ceedSetArray(CeedVector vec, CeedMemType mtype,
                              CeedCopyMode cmode, CeedScalar *array);

      static int ceedGetArray(CeedVector vec, CeedMemType mtype,
                              CeedScalar **array);

      static int ceedGetArrayRead(CeedVector vec, CeedMemType mtype,
                                  const CeedScalar **array);

      static int ceedRestoreArray(CeedVector vec, CeedScalar **array);

      static int ceedRestoreArrayRead(CeedVector vec, CeedScalar **array);

      static int ceedDestroy(CeedVector vec);
    };
  }
}

#endif
