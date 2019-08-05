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

#include <occa.hpp>

#include "composite-operator.hpp"
#include "elem-restriction.hpp"
#include "operator.hpp"
#include "qfunction.hpp"
#include "simplex-basis.hpp"
#include "tensor-basis.hpp"
#include "types.hpp"
#include "vector.hpp"


namespace ceed {
  namespace occa {
    std::string getDefaultDeviceMode(const bool cpuMode, const bool gpuMode) {
      // In case both cpuMode and gpuMode are set, prioritize the GPU if available
      // For example, if the resource is "/*/occa"
      if (gpuMode) {
        if (::occa::modeIsEnabled("CUDA")) {
          return "CUDA";
        }
        if (::occa::modeIsEnabled("HIP")) {
          return "HIP";
        }
        if (::occa::modeIsEnabled("OpenCL")) {
          return "OpenCL";
        }
        // Metal doesn't support doubles
      }

      if (cpuMode) {
        if (::occa::modeIsEnabled("OpenMP")) {
          return "OpenMP";
        }
        return "Serial";
      }

      return "";
    }

    void setDefaultProps(::occa::properties &deviceProps,
                         const std::string &defaultMode) {
      deviceProps["mode"] = defaultMode;

      // Set default device id
      if ((defaultMode == "CUDA")
          || (defaultMode == "HIP")
          || (defaultMode == "OpenCL")) {
        if (!deviceProps.has("device_id")) {
          deviceProps["device_id"] = 0;
        }
      }

      // Set default platform id
      if (defaultMode == "OpenCL") {
        if (!deviceProps.has("platform_id")) {
          deviceProps["platform_id"] = 0;
        }
      }
    }

    static int initCeed(const char *c_resource, Ceed ceed) {
      const std::string resource(c_resource);
      bool cpuMode = resource.find("/cpu/occa") != std::string::npos;
      bool gpuMode = resource.find("/gpu/occa") != std::string::npos;
      bool anyMode = resource.find("/*/occa") != std::string::npos;
      bool validResource = (cpuMode || gpuMode || anyMode);
      std::string resourceProps;

      // Make sure resource is an occa resource
      // Valid:
      //   /cpu/occa
      //   /cpu/occa/
      //   /*/occa/{mode:'CUDA',device_id:0}
      //   /gpu/occa/{mode:'CUDA',device_id:0}
      // Invalid:
      //   /cpu/occa-not
      if (validResource) {
        // Length of "/cpu/occa" and "/gpu/occa": 9
        // Length of "/*/occa": 7
        const size_t propsIndex = anyMode ? 7 : 9;

        if (resource.size() > propsIndex) {
          validResource = (resource[propsIndex] == '/');
          if (validResource && resource.size() > (propsIndex + 1)) {
            resourceProps = resource.substr(propsIndex + 1);
          }
        }
      }

      if (!validResource) {
        return CeedError(ceed, 1, "OCCA backend cannot use resource: %s", c_resource);
      }

      ::occa::properties deviceProps(resourceProps);
      if (!deviceProps.has("mode")) {
        const std::string defaultMode = (
          ceed::occa::getDefaultDeviceMode(cpuMode || anyMode,
                                           gpuMode || anyMode)
        );
        if (!defaultMode.size()) {
          return CeedError(ceed, 1,
                           "No available OCCA mode for the given resource: %s",
                           c_resource);
        }
        setDefaultProps(deviceProps, defaultMode);
      }

      int ierr;
      ceed::occa::Context *context = new Context();
      ierr = CeedSetData(ceed, (void**) &context); CeedChk(ierr);

      context->device = ::occa::device(deviceProps);

      return 0;
    }

    static int destroyCeed(Ceed ceed) {
      delete Context::from(ceed);
      return 0;
    }

    static int registerCeedFunction(Ceed ceed, const char *fname, ceed::occa::ceedFunction f) {
      return CeedSetBackendFunction(ceed, "Ceed", ceed, fname, f);
    }

    static int preferHostMemType(CeedMemType *type) {
      *type = CEED_MEM_HOST;
      return 0;
    }

    static int preferDeviceMemType(CeedMemType *type) {
      *type = CEED_MEM_DEVICE;
      return 0;
    }

    static ceed::occa::ceedFunction getPreferredMemType(Ceed ceed) {
      if (Context::from(ceed)->device.hasSeparateMemorySpace()) {
        return (ceed::occa::ceedFunction) preferDeviceMemType;
      }
      return (ceed::occa::ceedFunction) preferHostMemType;
    }

    static int registerMethods(Ceed ceed) {
      int ierr;

      ierr = registerCeedFunction(
        ceed, "Destroy",
        (ceed::occa::ceedFunction) ceed::occa::destroyCeed
      ); CeedChk(ierr);

      ierr = registerCeedFunction(
        ceed, "GetPreferredMemType",
        getPreferredMemType(ceed)
      ); CeedChk(ierr);

      ierr = registerCeedFunction(
        ceed, "VectorCreate",
        (ceed::occa::ceedFunction) ceed::occa::Vector::ceedCreate
      ); CeedChk(ierr);

      ierr = registerCeedFunction(
        ceed, "BasisCreateTensorH1",
        (ceed::occa::ceedFunction) ceed::occa::TensorBasis::ceedCreate
      ); CeedChk(ierr);

      ierr = registerCeedFunction(
        ceed, "BasisCreateH1",
        (ceed::occa::ceedFunction) ceed::occa::SimplexBasis::ceedCreate
      ); CeedChk(ierr);

      ierr = registerCeedFunction(
        ceed, "ElemRestrictionCreate",
        (ceed::occa::ceedFunction) ceed::occa::ElemRestriction::ceedCreate
      ); CeedChk(ierr);

      ierr = registerCeedFunction(
        ceed, "ElemRestrictionCreateBlocked",
        (ceed::occa::ceedFunction) ceed::occa::ElemRestriction::ceedCreate
      ); CeedChk(ierr);

      ierr = registerCeedFunction(
        ceed, "QFunctionCreate",
        (ceed::occa::ceedFunction) ceed::occa::QFunction::ceedCreate
      ); CeedChk(ierr);

      ierr = registerCeedFunction(
        ceed, "OperatorCreate",
        (ceed::occa::ceedFunction) ceed::occa::Operator::ceedCreate
      ); CeedChk(ierr);

      ierr = registerCeedFunction(
        ceed, "CompositeOperatorCreate",
        (ceed::occa::ceedFunction) ceed::occa::CompositeOperator::ceedCreate
      ); CeedChk(ierr);

      return 0;
    }

    static int registerBackend(const char *resource, Ceed ceed) {
      int ierr;

      try {
        ierr = ceed::occa::initCeed(resource, ceed); CeedChk(ierr);
        ierr = ceed::occa::registerMethods(ceed); CeedChk(ierr);
      } catch (::occa::exception exc) {
        std::string error = exc.toString();
        return CeedError(ceed, 1, error.c_str());
      }

      return 0;
    }
  }
}

__attribute__((constructor))
static void Register(void) {
  CeedRegister("/*/occa", ceed::occa::registerBackend, 20);
  CeedRegister("/gpu/occa", ceed::occa::registerBackend, 20);
  CeedRegister("/cpu/occa", ceed::occa::registerBackend, 20);
}
