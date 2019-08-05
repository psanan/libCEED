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

#ifndef CEED_OCCA_KERNELS_HEADER_HEADER
#define CEED_OCCA_KERNELS_HEADER_HEADER

// Based on cuda-shared kernels
//
// Expects the following types:
// - CeedInt
// - CeedScalar
//
// Expects the following definitions:
// - Q1D                  : CeedInt
// - P1D                  : CeedInt
// - ELEMENTS_PER_BLOCK   : CeedInt
// - BASIS_COMPONENT_COUNT: CeedInt
// - SHARED_BUFFER_SIZE   : CeedInt
//       1D: (Q1D * ELEMENTS_PER_BLOCK)
//       2D: (Q1D * Q1D * ELEMENTS_PER_BLOCK)
//       3D: (Q1D * Q1D * ELEMENTS_PER_BLOCK)

#define STRINGIFY_SOURCE(...) #__VA_ARGS__

#endif
