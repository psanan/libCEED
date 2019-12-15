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

#include "./elem-restriction.hpp"
#include "./vector.hpp"
#include "./kernels/elem-restriction.okl"

namespace ceed {
  namespace occa {
    ElemRestriction::ElemRestriction() :
        ceed(NULL),
        ceedElementCount(0),
        ceedElementSize(0),
        ceedNodeCount(0),
        ceedComponentCount(0),
        ceedBlockSize(0),
        freeHostIndices(true),
        hostIndices(NULL),
        freeIndices(true) {}

    ElemRestriction::~ElemRestriction() {
      applyWithVTransposeKernelBuilder.free();
      applyWithoutVTransposeKernelBuilder.free();

      if (freeHostIndices) {
        CeedFree(&hostIndices);
      }

      if (freeIndices) {
        indices.free();
      }
      transposeOffsets.free();
      transposeIndices.free();
    }

    void ElemRestriction::setup(CeedMemType memType,
                                CeedCopyMode copyMode,
                                const CeedInt *indicesInput) {
      if (memType == CEED_MEM_HOST) {
        setupFromHostMemory(copyMode, indicesInput);
      } else {
        setupFromDeviceMemory(copyMode, indicesInput);
      }

      setupKernelBuilders();
    }

    void ElemRestriction::setupFromHostMemory(CeedCopyMode copyMode,
                                              const CeedInt *indices_h) {
      freeHostIndices = (copyMode == CEED_OWN_POINTER);

      if ((copyMode == CEED_OWN_POINTER) || (copyMode == CEED_USE_POINTER)) {
        hostIndices = const_cast<CeedInt*>(indices_h);
      }

      if (hostIndices) {
        indices = getDevice().malloc<CeedInt>(ceedElementCount * ceedElementSize,
                                              hostIndices);
      }
    }

    void ElemRestriction::setupFromDeviceMemory(CeedCopyMode copyMode,
                                                const CeedInt *indices_d) {
      ::occa::memory deviceIndices = arrayToMemory((CeedScalar*) indices_d);

      freeIndices = (copyMode == CEED_OWN_POINTER);

      if (copyMode == CEED_COPY_VALUES) {
        indices = deviceIndices.clone();
      } else {
        indices = deviceIndices;
      }
    }

    void ElemRestriction::setupKernelBuilders() {
      ::occa::properties kernelProps;
      kernelProps["defines/CeedInt"]    = ::occa::dtype::get<CeedInt>().name();
      kernelProps["defines/CeedScalar"] = ::occa::dtype::get<CeedScalar>().name();

      kernelProps["defines/COMPONENT_COUNT"] = ceedComponentCount;
      kernelProps["defines/ELEMENT_SIZE"]    = ceedElementSize;
      kernelProps["defines/NODE_COUNT"]      = ceedNodeCount;
      kernelProps["defines/TILE_SIZE"]       = 64;
      kernelProps["defines/USES_INDICES"]    = indices.isInitialized();

      applyWithVTransposeKernelBuilder = ::occa::kernelBuilder::fromString(
        elemRestriction_source, "applyWithVTranspose", kernelProps
      );

      applyWithoutVTransposeKernelBuilder = ::occa::kernelBuilder::fromString(
        elemRestriction_source, "applyWithoutVTranspose", kernelProps
      );
    }

    void ElemRestriction::setupTransposeIndices() {
      if (transposeOffsets.isInitialized()) {
        return;
      }

      if (hostIndices) {
        setupTransposeIndices(hostIndices);
      } else {
        // Use a temporary buffer to compute transpose indices
        CeedInt *indices_h = new CeedInt[indices.length()];
        indices.copyTo((void*) indices_h);

        setupTransposeIndices(indices_h);

        delete [] indices_h;
      }
    }

    void ElemRestriction::setupTransposeIndices(const CeedInt *indices_h) {
      const CeedInt offsetsCount = ceedNodeCount + 1;
      const CeedInt elementEntryCount = ceedElementCount * ceedElementSize;

      CeedInt *transposeOffsets_h = new CeedInt[offsetsCount];
      CeedInt *transposeIndices_h = new CeedInt[elementEntryCount];

      // Setup offsets
      for (CeedInt i = 0; i < offsetsCount; ++i) {
        transposeOffsets_h[i] = 0;
      }
      for (CeedInt i = 0; i < elementEntryCount; ++i) {
        ++transposeOffsets_h[indices_h[i] + 1];
      }
      for (CeedInt i = 1; i < offsetsCount; ++i) {
        transposeOffsets_h[i] += transposeOffsets_h[i - 1];
      }

      // Setup indices
      for (CeedInt i = 0; i < elementEntryCount; ++i) {
        const CeedInt index = transposeOffsets_h[indices_h[i]]++;
        transposeIndices_h[index] = i;
      }

      // Reset offsets
      for (int i = offsetsCount - 1; i > 0; --i) {
        transposeOffsets_h[i] = transposeOffsets_h[i - 1];
      }
      transposeOffsets_h[0] = 0;

      // Copy to device
      transposeOffsets = getDevice().malloc<CeedInt>(offsetsCount,
                                                     transposeOffsets_h);
      transposeIndices = getDevice().malloc<CeedInt>(elementEntryCount,
                                                     transposeIndices_h);
    }

    ElemRestriction* ElemRestriction::from(CeedElemRestriction r) {
      if (!r) {
        return NULL;
      }

      int ierr;
      ElemRestriction *elemRestriction;

      ierr = CeedElemRestrictionGetData(r, (void**) &elemRestriction); CeedOccaFromChk(ierr);
      ierr = CeedElemRestrictionGetCeed(r, &elemRestriction->ceed); CeedOccaFromChk(ierr);

      ierr = CeedElemRestrictionGetNumElements(r, &elemRestriction->ceedElementCount);
      CeedOccaFromChk(ierr);

      ierr = CeedElemRestrictionGetElementSize(r, &elemRestriction->ceedElementSize);
      CeedOccaFromChk(ierr);

      ierr = CeedElemRestrictionGetNumNodes(r, &elemRestriction->ceedNodeCount);
      CeedOccaFromChk(ierr);

      ierr = CeedElemRestrictionGetNumComponents(r, &elemRestriction->ceedComponentCount);
      CeedOccaFromChk(ierr);

      ierr = CeedElemRestrictionGetBlockSize(r, &elemRestriction->ceedBlockSize);
      CeedOccaFromChk(ierr);

      return elemRestriction;
    }

    ElemRestriction* ElemRestriction::from(CeedOperatorField operatorField) {
      int ierr;
      CeedElemRestriction ceedElemRestriction;

      ierr = CeedOperatorFieldGetElemRestriction(operatorField, &ceedElemRestriction);
      CeedOccaFromChk(ierr);

      return from(ceedElemRestriction);
    }

    ::occa::device ElemRestriction::getDevice() {
      return Context::from(ceed)->device;
    }

    ::occa::kernel ElemRestriction::buildApplyKernel(const bool uIsTransposed,
                                                     const bool vIsTransposed) {
      ::occa::properties kernelProps;
      kernelProps["defines/U_IS_TRANSPOSED"] = uIsTransposed;

      return (
        vIsTransposed
        ? applyWithVTransposeKernelBuilder.build(getDevice(), kernelProps)
        : applyWithoutVTransposeKernelBuilder.build(getDevice(), kernelProps)
      );
    }

    int ElemRestriction::apply(CeedTransposeMode vTransposeMode,
                               CeedTransposeMode uTransposeMode,
                               Vector &u,
                               Vector &v,
                               CeedRequest *request) {
      const bool uIsTransposed = (uTransposeMode != CEED_NOTRANSPOSE);
      const bool vIsTransposed = (vTransposeMode != CEED_NOTRANSPOSE);

      ::occa::kernel apply = buildApplyKernel(uIsTransposed, vIsTransposed);

      if (vIsTransposed) {
        setupTransposeIndices();
        apply(transposeOffsets,
              transposeIndices,
              u.getConstKernelArg(),
              v.getKernelArg());
      } else {
        apply(ceedElementCount,
              indices,
              u.getConstKernelArg(),
              v.getKernelArg());
      }

      return 0;
    }

    int ElemRestriction::applyBlock(CeedInt block,
                                    CeedTransposeMode vTransposeMode,
                                    CeedTransposeMode uTransposeMode,
                                    Vector &u,
                                    Vector &v,
                                    CeedRequest *request) {
      // TODO: Implement
      return CeedError(ceed, 1, "Block apply not supported yet");
    }

    //---[ Ceed Callbacks ]-----------
    int ElemRestriction::registerRestrictionFunction(Ceed ceed, CeedElemRestriction r,
                                                     const char *fname, ceed::occa::ceedFunction f) {
      return CeedSetBackendFunction(ceed, "ElemRestriction", r, fname, f);
    }

    int ElemRestriction::ceedCreate(CeedMemType memType,
                                    CeedCopyMode copyMode,
                                    const CeedInt *indicesInput,
                                    CeedElemRestriction r) {
      int ierr;
      Ceed ceed;
      ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);

      if ((memType != CEED_MEM_DEVICE) && (memType != CEED_MEM_HOST)) {
        return CeedError(ceed, 1, "Only HOST and DEVICE CeedMemType supported");
      }

      ElemRestriction *elemRestriction = new ElemRestriction();
      ierr = CeedElemRestrictionSetData(r, (void**) &elemRestriction); CeedChk(ierr);

      // Setup Ceed objects before setting up memory
      elemRestriction = ElemRestriction::from(r);
      elemRestriction->setup(memType, copyMode, indicesInput);

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
