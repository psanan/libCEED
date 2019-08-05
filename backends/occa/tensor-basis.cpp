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

#include "tensor-basis.hpp"
#include "kernels/tensor-basis-1d.okl"
#include "kernels/tensor-basis-2d.okl"
#include "kernels/tensor-basis-3d.okl"


namespace ceed {
  namespace occa {
    TensorBasis::TensorBasis(CeedBasis basis,
                             CeedInt dim_,
                             CeedInt P1D_,
                             CeedInt Q1D_,
                             const CeedScalar *interp1D_,
                             const CeedScalar *grad1D_,
                             const CeedScalar *qWeight1D_) :
        dim(dim_),
        P1D(P1D_),
        Q1D(Q1D_) {
      setCeedFields(basis);

      ::occa::device device = getDevice();

      ::occa::dtype_t ceedScalarDtype = ::occa::dtype::get<CeedScalar>();
      interp1D  = device.malloc(P1D, ceedScalarDtype, interp1D_);
      grad1D    = device.malloc(P1D, ceedScalarDtype, grad1D_);
      qWeight1D = device.malloc(Q1D, ceedScalarDtype, qWeight1D_);

      const char *kernelSource = NULL;
      std::string sharedBufferSize;
      if (dim == 1) {
        kernelSource = tensorBasis1DSource;
        sharedBufferSize = "(Q1D * ELEMENTS_PER_BLOCK)";
      } else if (dim == 2) {
        kernelSource = tensorBasis2DSource;
        sharedBufferSize = "(Q1D * Q1D * BASIS_COMPONENT_COUNT * ELEMENTS_PER_BLOCK)";
      } else {
        kernelSource = tensorBasis3DSource;
        sharedBufferSize = "(Q1D * Q1D * BASIS_COMPONENT_COUNT * ELEMENTS_PER_BLOCK)";
      }

      ::occa::properties kernelProps;
      kernelProps["defines/CeedInt"]    = ::occa::dtype::get<CeedInt>().name();
      kernelProps["defines/CeedScalar"] = ::occa::dtype::get<CeedScalar>().name();
      kernelProps["defines/Q1D"] = Q1D;
      kernelProps["defines/P1D"] = P1D;
      kernelProps["defines/BASIS_COMPONENT_COUNT"] = ceedComponentCount;

      interpKernelBuilder = ::occa::kernelBuilder::fromString(
        kernelSource, "interp", kernelProps
      );
      gradKernelBuilder = ::occa::kernelBuilder::fromString(
        kernelSource, "grad"  , kernelProps
      );
      weightKernelBuilder = ::occa::kernelBuilder::fromString(
        kernelSource, "weight", kernelProps
      );
    }

    TensorBasis::~TensorBasis() {
      interpKernelBuilder.free();
      gradKernelBuilder.free();
      weightKernelBuilder.free();
      interp1D.free();
      grad1D.free();
      qWeight1D.free();
    }

    ::occa::device TensorBasis::getDevice() {
      return Context::from(ceed)->device;
    }

    int TensorBasis::applyInterp(const CeedInt elementCount,
                                 const bool transpose,
                                 Vector &U,
                                 Vector &V) {
      ::occa::kernel interp = getInterpKernel();
      interp(elementCount,
             transpose,
             interp1D,
             U.getConstKernelArg(),
             V.getKernelArg());
      return 0;
    }

    ::occa::kernel TensorBasis::getInterpKernel() {
      int elementsPerBlock;
      int sharedBufferSize;
      if (dim == 1) {
        elementsPerBlock = 32;
        sharedBufferSize = Q1D * elementsPerBlock;
      } else if (dim == 2) {
        const CeedInt blocksByQ[7] = {0, 32, 8, 6, 4, 2, 8};
        if (Q1D < 7) {
          elementsPerBlock = blocksByQ[Q1D];
        } else {
          elementsPerBlock = 1;
        }
        sharedBufferSize = Q1D * Q1D * ceedComponentCount * elementsPerBlock;
      } else {
        elementsPerBlock = 1;
        sharedBufferSize = Q1D * Q1D * ceedComponentCount * elementsPerBlock;
      }

      return buildEvalKernel(interpKernelBuilder, elementsPerBlock, sharedBufferSize);
    }

    int TensorBasis::applyGrad(const CeedInt elementCount,
                               const bool transpose,
                               Vector &U,
                               Vector &V) {
      ::occa::kernel grad = getGradKernel();
      grad(elementCount,
           transpose,
           interp1D, grad1D,
           U.getConstKernelArg(),
           V.getKernelArg());
      return 0;
    }

    ::occa::kernel TensorBasis::getGradKernel() {
      int elementsPerBlock;
      int sharedBufferSize;
      if (dim == 1) {
        elementsPerBlock = 32;
        sharedBufferSize = Q1D * elementsPerBlock;
      } else if (dim == 2) {
        const CeedInt blocksByQ[7] = {0, 32, 8, 6, 4, 2, 8};
        if (Q1D < 7) {
          elementsPerBlock = blocksByQ[Q1D];
        } else {
          elementsPerBlock = 1;
        }
        sharedBufferSize = Q1D * Q1D * ceedComponentCount * elementsPerBlock;
      } else {
        elementsPerBlock = 1;
        sharedBufferSize = Q1D * Q1D * ceedComponentCount * elementsPerBlock;
      }

      return buildEvalKernel(gradKernelBuilder, elementsPerBlock, sharedBufferSize);
    }

    int TensorBasis::applyWeight(const CeedInt elementCount,
                                 Vector &W) {
      ::occa::kernel weight = getWeightKernel();
      weight(elementCount, qWeight1D, W.getKernelArg());
      return 0;
    }

    ::occa::kernel TensorBasis::getWeightKernel() {
      int elementsPerBlock;
      if (dim == 1) {
        elementsPerBlock = 32 / Q1D;
      } else if (dim == 2) {
        if ((Q1D * Q1D) > 32) {
          elementsPerBlock = 1;
        } else {
          elementsPerBlock = 32 / (Q1D * Q1D);
        }
      } else {
        elementsPerBlock = Q1D;
      }

      return buildEvalKernel(weightKernelBuilder, elementsPerBlock, 1);
    }

    ::occa::kernel TensorBasis::buildEvalKernel(::occa::kernelBuilder &kernelBuilder,
                                                const int elementsPerBlock,
                                                const int sharedBufferSize) {

      ::occa::properties kernelProps;
      kernelProps["defines/ELEMENTS_PER_BLOCK"] = elementsPerBlock;
      kernelProps["defines/SHARED_BUFFER_SIZE"] = sharedBufferSize;

      return kernelBuilder.build(getDevice(), kernelProps);
    }

    int TensorBasis::apply(const CeedInt elementCount,
                           CeedTransposeMode tmode,
                           CeedEvalMode emode,
                           Vector *U,
                           Vector *V) {
      const bool transpose = tmode == CEED_TRANSPOSE;

      if ((dim < 1) || (3 < dim)) {
        return CeedError(
          NULL, 1,
          "Backend only supports dimensions 1, 2, and 3. Given: %i", dim
        );
      }

      // Check arguments
      switch (emode) {
        case CEED_EVAL_INTERP:
        case CEED_EVAL_GRAD:
          if (!U) {
            return CeedError(NULL, 1, "Incorrect CeedVector input: U");
          }
          if (!V) {
            return CeedError(NULL, 1, "Incorrect CeedVector input: V");
          }
          break;
        case CEED_EVAL_WEIGHT:
          if (!V) {
            return CeedError(NULL, 1, "Incorrect CeedVector input: V");
          }
          break;
        default:
          return CeedError(NULL, 1, "Backend does not support eval mode: %i", (int) emode);
      }

      try {
        // Apply kernel
        switch (emode) {
          case CEED_EVAL_INTERP:
            return applyInterp(elementCount, transpose, *U, *V);
          case CEED_EVAL_GRAD:
            return applyGrad(elementCount, transpose, *U, *V);
          case CEED_EVAL_WEIGHT:
            return applyWeight(elementCount, *V);
          default: {}
        }
      } catch (::occa::exception exc) {
        // Handle kernel build errors the CEED way
        CeedHandleOccaException(exc);
      }

      return 0;
    }

    //---[ Ceed Callbacks ]-------------
    int TensorBasis::ceedCreate(CeedInt dim,
                                CeedInt P1D, CeedInt Q1D,
                                const CeedScalar *interp1D,
                                const CeedScalar *grad1D,
                                const CeedScalar *qref1D,
                                const CeedScalar *qWeight1D,
                                CeedBasis basis) {
      // Based on cuda-shared
      if (Q1D < P1D) {
        return CeedError(
          NULL, 1,
          "Backend does not support underintegrated basis (P1D: %i, Q1D: %1)",
          (int) P1D, (int) Q1D
        );
      }

      int ierr;
      Ceed ceed;
      ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);

      TensorBasis *basis_ = new TensorBasis(basis,
                                            dim,
                                            P1D, Q1D,
                                            interp1D, grad1D, qWeight1D);
      ierr = CeedBasisSetData(basis, (void**) &basis_); CeedChk(ierr);

      ierr = registerBasisFunction(ceed, basis, "Apply",
                                   (ceed::occa::ceedFunction) Basis::ceedApply);
      CeedChk(ierr);

      ierr = registerBasisFunction(ceed, basis, "Destroy",
                                   (ceed::occa::ceedFunction) Basis::ceedDestroy);
      CeedChk(ierr);

      return 0;
    }
  }
}
