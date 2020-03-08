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

#include <map>

#include "./elem-restriction.hpp"
#include "./vector.hpp"
#include "./kernels/elem-restriction.okl"

namespace ceed {
  namespace occa {
    ElemRestriction::ElemRestriction() :
        ceedElementCount(0),
        ceedElementSize(0),
        ceedNodeCount(0),
        ceedComponentCount(0),
        ceedBlockSize(0),
        ceedInterlaceMode(CEED_NONINTERLACED),
        freeHostIndices(true),
        hostIndices(NULL),
        freeIndices(true) {}

    ElemRestriction::~ElemRestriction() {
      if (freeHostIndices) {
        CeedFree(&hostIndices);
      }
      if (freeIndices) {
        indices.free();
      }
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

      applyWithRTransposeKernelBuilder = ::occa::kernelBuilder::fromString(
        elemRestriction_source, "applyWithRTranspose", kernelProps
      );

      applyWithoutRTransposeKernelBuilder = ::occa::kernelBuilder::fromString(
        elemRestriction_source, "applyWithoutRTranspose", kernelProps
      );

      // Add block size
      kernelProps["defines/BLOCK_SIZE"] = ceedBlockSize;

      applyBlockedWithRTransposeKernelBuilder = ::occa::kernelBuilder::fromString(
        elemRestrictionBlocked_source, "applyWithRTranspose", kernelProps
      );

      applyBlockedWithoutRTransposeKernelBuilder = ::occa::kernelBuilder::fromString(
        elemRestrictionBlocked_source, "applyWithoutRTranspose", kernelProps
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
      ::occa::device device = getDevice();

      transposeOffsets = device.malloc<CeedInt>(offsetsCount,
                                                transposeOffsets_h);
      transposeIndices = device.malloc<CeedInt>(elementEntryCount,
                                                transposeIndices_h);
    }

    void ElemRestriction::setupTransposeBlockIndices() {
      if (transposeOffsets.isInitialized()) {
        return;
      }

      if (hostIndices) {
        setupTransposeBlockIndices(hostIndices);
      } else {
        // Use a temporary buffer to compute transpose indices
        CeedInt *indices_h = new CeedInt[indices.length()];
        indices.copyTo((void*) indices_h);

        setupTransposeBlockIndices(indices_h);

        delete [] indices_h;
      }
    }

    void ElemRestriction::setupTransposeBlockIndices(const CeedInt *indices_h) {
      std::vector<CeedInt> uOffsets, uIndices;
      std::vector<CeedInt> vOffsets, vIndices;

      uOffsets.push_back(0);
      vOffsets.push_back(0);

      for (int blockOffset = 0; blockOffset < ceedElementCount; blockOffset += ceedBlockSize) {
        const int lastBlockElement = std::min(ceedBlockSize, ceedElementCount - blockOffset);

        // Store element in charge of updating a given v node
        std::map<int, int> vIndexElement;

        // Store u nodes for a given v node
        std::map<int, std::vector<int>> uIndicesMap;

        // Store which element is in change of updating a given vIndex
        std::vector<int> *elementToVIndex = new std::vector<int>[ceedBlockSize];

        for (int blockElement = 0; blockElement < lastBlockElement; ++blockElement) {
          for (int n = 0; n < ceedElementSize; ++n) {
            const int indexOffset = blockElement + (n * ceedBlockSize);

            const int uIndex = (
              indexOffset
            );
            const int vIndex = indices_h[
              (blockOffset * ceedElementSize)
              + indexOffset
            ];

            vIndexElement[vIndex] = blockElement;
            uIndicesMap[vIndex].push_back(uIndex);
          }
        }

        // Transpose vIndex -> blockElement map
        for (auto it = vIndexElement.begin(); it != vIndexElement.end(); ++it) {
          const int vIndex = it->first;
          const int ownerElement = it->second;
          elementToVIndex[ownerElement].push_back(vIndex);
        }

        for (int blockElement = 0; blockElement < lastBlockElement; ++blockElement) {
          const std::vector<int> &elementVIndices = elementToVIndex[blockElement];
          const int vIndicesCount = (int) elementVIndices.size();

          vOffsets.push_back(vOffsets.back() + vIndicesCount);

          for (int vi = 0; vi < vIndicesCount; ++vi) {
            const int vIndex = elementVIndices[vi];
            const std::vector<int> &elementUIndices = uIndicesMap[vIndex];
            const int uIndicesCount = (int) elementUIndices.size();

            vIndices.push_back(vIndex);
            uOffsets.push_back(uOffsets.back() + uIndicesCount);

            for (int ui = 0; ui < uIndicesCount; ++ui) {
              uIndices.push_back(elementUIndices[ui]);
            }
          }
        }

        delete [] elementToVIndex;
      }

      // Copy to device
      ::occa::device device = getDevice();

      blockedTransposeUOffsets = device.malloc<CeedInt>(uOffsets.size(), uOffsets.data());
      blockedTransposeVOffsets = device.malloc<CeedInt>(vOffsets.size(), vOffsets.data());
      blockedTransposeUIndices = device.malloc<CeedInt>(uIndices.size(), uIndices.data());
      blockedTransposeVIndices = device.malloc<CeedInt>(vIndices.size(), vIndices.data());
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

      elemRestriction->ceedInterlaceMode = CEED_NONINTERLACED;
      if (elemRestriction->hostIndices) {
        ierr = CeedElemRestrictionGetIMode(r, &elemRestriction->ceedInterlaceMode);
        CeedOccaFromChk(ierr);
      }

      // Set to at least 1
      elemRestriction->ceedBlockSize = std::max(1, elemRestriction->ceedBlockSize);

      return elemRestriction;
    }

    ElemRestriction* ElemRestriction::from(CeedOperatorField operatorField) {
      int ierr;
      CeedElemRestriction ceedElemRestriction;

      ierr = CeedOperatorFieldGetElemRestriction(operatorField, &ceedElemRestriction);
      CeedOccaFromChk(ierr);

      return from(ceedElemRestriction);
    }

    ::occa::kernel ElemRestriction::buildApplyKernel(const bool rIsTransposed,
                                                     const bool compIsFastIndex) {
      ::occa::properties kernelProps;
      kernelProps["defines/COMP_IS_FAST_INDEX"] = compIsFastIndex;

      return (
        rIsTransposed
        ? applyWithRTransposeKernelBuilder.build(getDevice(), kernelProps)
        : applyWithoutRTransposeKernelBuilder.build(getDevice(), kernelProps)
      );
    }

    int ElemRestriction::apply(CeedTransposeMode rTransposeMode,
                               Vector &u,
                               Vector &v,
                               CeedRequest *request) {
      const bool rIsTransposed = (rTransposeMode != CEED_NOTRANSPOSE);
      const bool compIsFastIndex = (ceedInterlaceMode != CEED_NONINTERLACED);

      ::occa::kernel apply = buildApplyKernel(rIsTransposed, compIsFastIndex);

      if (rIsTransposed) {
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

    ::occa::kernel ElemRestriction::buildApplyBlockedKernel(const bool rIsTransposed,
                                                            const bool compIsFastIndex) {
      ::occa::properties kernelProps;
      kernelProps["defines/COMP_IS_FAST_INDEX"] = compIsFastIndex;

      return (
        rIsTransposed
        ? applyBlockedWithRTransposeKernelBuilder.build(getDevice(), kernelProps)
        : applyBlockedWithoutRTransposeKernelBuilder.build(getDevice(), kernelProps)
      );
    }

    int ElemRestriction::applyBlock(CeedInt block,
                                    CeedTransposeMode rTransposeMode,
                                    Vector &u,
                                    Vector &v,
                                    CeedRequest *request) {
      const bool rIsTransposed = (rTransposeMode != CEED_NOTRANSPOSE);
      const bool compIsFastIndex = (ceedInterlaceMode != CEED_NONINTERLACED);

      ::occa::kernel apply = buildApplyBlockedKernel(rIsTransposed, compIsFastIndex);

      const int firstElement = block * ceedBlockSize;
      const int lastElement = std::min(ceedElementCount,
                                       (CeedInt) (firstElement + ceedBlockSize));

      if (rIsTransposed) {
        setupTransposeBlockIndices();

        apply(firstElement,
              lastElement,
              blockedTransposeUOffsets,
              blockedTransposeVOffsets,
              blockedTransposeUIndices,
              blockedTransposeVIndices,
              u.getConstKernelArg(),
              v.getKernelArg());
      } else {
        apply(firstElement,
              lastElement,
              indices,
              u.getConstKernelArg(),
              v.getKernelArg());
      }

      return 0;
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

    int ElemRestriction::ceedApply(CeedElemRestriction r, CeedTransposeMode tmode,
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

      return elemRestriction->apply(tmode, *uVector, *vVector, request);
    }

    int ElemRestriction::ceedApplyBlock(CeedElemRestriction r,
                                        CeedInt block, CeedTransposeMode tmode,
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

      return elemRestriction->applyBlock(block, tmode, *uVector, *vVector, request);
    }

    int ElemRestriction::ceedDestroy(CeedElemRestriction r) {
      delete ElemRestriction::from(r);
      return 0;
    }
  }
}
