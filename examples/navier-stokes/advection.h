// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
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

/// @file
/// Advection initial condition and operator for Navier-Stokes example using PETSc

#ifndef advection_h
#define advection_h

#ifndef CeedPragmaOMP
#  ifdef _OPENMP
#    define CeedPragmaOMP_(a) _Pragma(#a)
#    define CeedPragmaOMP(a) CeedPragmaOMP_(omp a)
#  else
#    define CeedPragmaOMP(a)
#  endif
#endif

#include <math.h>

// *****************************************************************************
// This QFunction sets the the initial conditions and boundary conditions
//
// Initial Conditions:
//   Mass Density:
//     Constant mass density of 1.0
//   Momentum Density:
//     Rotational field in x,y with no momentum in z
//   Energy Density:
//     Maximum of 1. x0 decreasing linearly to 0. as radial distance increases
//       to (1.-r/rc), then 0. everywhere else
//
//  Boundary Conditions:
//    Mass Density:
//      0.0 flux
//    Momentum Density:
//      0.0
//    Energy Density:
//      0.0 flux
//
// *****************************************************************************
static int ICsAdvection(void *ctx, CeedInt Q,
                        const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar (*X)[Q] = (CeedScalar(*)[Q])in[0];
  // Outputs
  CeedScalar (*q0)[Q] = (CeedScalar(*)[Q])out[0],
             (*coords)[Q] = (CeedScalar(*)[Q])out[1];
  // Context
  const CeedScalar *context = (const CeedScalar*)ctx;
  const CeedScalar rc = context[8];
  const CeedScalar lx = context[9];
  const CeedScalar ly = context[10];
  const CeedScalar lz = context[11];
  // Setup
  const CeedScalar tol = 1.e-14;
  const CeedScalar x0[3] = {0.25*lx, 0.5*ly, 0.5*lz};
  const CeedScalar center[3] = {0.5*lx, 0.5*ly, 0.5*lz};

  CeedPragmaOMP(simd)
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Coordinates
    const CeedScalar x = X[0][i];
    const CeedScalar y = X[1][i];
    const CeedScalar z = X[2][i];
    // -- Energy
    const CeedScalar r = sqrt(pow((x - x0[0]), 2) +
                              pow((y - x0[1]), 2) +
                              pow((z - x0[2]), 2));
//                              0);
// line above uncommented make a half cylinder  flip comment with 2 lines above to get  half sphere

    // Initial Conditions
    q0[0][i] = 1.;
    q0[1][i] = -0.5*(y - center[1]);
    q0[2][i] =  0.5*(x - center[0]);
    q0[3][i] = 0.0;
//    q0[4][i] = r <= rc ? (1.-r/rc) : 0.;
    q0[4][i] = ((r <= rc) && (y<center[1])) ? (1.-r/rc) : 0.;
// line above adds a conditional to cut the shape in half. 2 lines above is original smooth, full round
    // Homogeneous Dirichlet Boundary Conditions for Momentum
    if(1) {
    if ( fabs(x - 0.0) < tol || fabs(x - lx) < tol
         || fabs(y - 0.0) < tol || fabs(y - ly) < tol
         || fabs(z - 0.0) < tol || fabs(z - lz) < tol ) {
      q0[1][i] = 0.0;
      q0[2][i] = 0.0;
      q0[3][i] = 0.0;
    }
}

    // Coordinates
    coords[0][i] = x;
    coords[1][i] = y;
    coords[2][i] = z;

  } // End of Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
// This QFunction implements the following formulation of the advection equation
//
// This is 3D advection given in two formulations based upon the weak form.
//
// State Variables: q = ( rho, U1, U2, U3, E )
//   rho - Mass Density
//   Ui  - Momentum Density    ,  Ui = rho ui
//   E   - Total Energy Density
//
// Advection Equation:
//   dE/dt + div( E u ) = 0
//
// *****************************************************************************
static int Advection(void *ctx, CeedInt Q,
                     const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar (*q)[Q] = (CeedScalar(*)[Q])in[0],
                   (*dq)[5][Q] = (CeedScalar(*)[5][Q])in[1],
                   (*qdata)[Q] = (CeedScalar(*)[Q])in[2],
                   (*x)[Q] = (CeedScalar(*)[Q])in[3];
  // Outputs
  CeedScalar (*v)[Q] = (CeedScalar(*)[Q])out[0],
             (*dv)[5][Q] = (CeedScalar(*)[5][Q])out[1];

  // Context
  const CeedScalar *context = (const CeedScalar*)ctx;
  const CeedScalar CtauS      = context[0];

  CeedPragmaOMP(simd)
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Interp in
    const CeedScalar rho        =    q[0][i];
    const CeedScalar u[3]       =   {q[1][i] / rho,
                                     q[2][i] / rho,
                                     q[3][i] / rho
                                    };
    const CeedScalar E          =    q[4][i];
    // -- Grad in
    const CeedScalar drho[3]    =   {dq[0][0][i],
                                     dq[1][0][i],
                                     dq[2][0][i]
                                    };
    const CeedScalar du[3][3]   = {{(dq[0][1][i] - drho[0]*u[0]) / rho,
                                    (dq[1][1][i] - drho[1]*u[0]) / rho,
                                    (dq[2][1][i] - drho[2]*u[0]) / rho},
                                   {(dq[0][2][i] - drho[0]*u[1]) / rho,
                                    (dq[1][2][i] - drho[1]*u[1]) / rho,
                                    (dq[2][2][i] - drho[2]*u[1]) / rho},
                                   {(dq[0][3][i] - drho[0]*u[2]) / rho,
                                    (dq[1][3][i] - drho[1]*u[2]) / rho,
                                    (dq[2][3][i] - drho[2]*u[2]) / rho}
                                  };
    const CeedScalar dE[3]      =   {dq[0][4][i],
                                     dq[1][4][i],
                                     dq[2][4][i]
                                    };
    // -- Interp-to-Interp qdata
    const CeedScalar wJ         =    qdata[0][i];
    // -- Interp-to-Grad qdata
    // ---- Inverse of change of coordinate matrix: X_i,j
    const CeedScalar dXdx[3][3] =  {{qdata[1][i],
                                     qdata[2][i],
                                     qdata[3][i]},
                                    {qdata[4][i],
                                     qdata[5][i],
                                     qdata[6][i]},
                                    {qdata[7][i],
                                     qdata[8][i],
                                     qdata[9][i]}
                                   };

    // The Physics

    // -- Density
    // ---- No Change
    dv[0][0][i] = 0;
    dv[1][0][i] = 0;
    dv[2][0][i] = 0;
    v[0][1] = 0;

    // -- Momentum
    // ---- No Change
    dv[0][1][i] = 0;
    dv[1][1][i] = 0;
    dv[2][1][i] = 0;
    dv[0][2][i] = 0;
    dv[1][2][i] = 0;
    dv[2][2][i] = 0;
    dv[0][3][i] = 0;
    dv[1][3][i] = 0;
    dv[2][3][i] = 0;
    v[1][i] = 0;
    v[2][i] = 0;
    v[3][i] = 0;

    // -- Total Energy
    CeedInt IBP=1; // for now we are setting this to 1 but we could make it a run time arg.  Switch to 0 to get non-IBP on Galerkin term
    CeedScalar divConv;
    if(IBP==0 || CtauS > 0){ // only compute if needed
       CeedScalar div_u = 0, u_dot_grad_E = 0;
       for (int j=0; j<3; j++) {
         CeedScalar dEdx_j = 0;
         for (int k=0; k<3; k++) {
           div_u += du[k][j] * dXdx[k][j]; // u_{j,j} = u_{j,K} X_{K,j}
           dEdx_j += dE[k] * dXdx[k][j];
         }
         u_dot_grad_E += u[j] * dEdx_j;
       }
       divConv = -wJ * (E*div_u + u_dot_grad_E);
    } 
    switch (IBP) {
    // ---- Version 1: dv \cdot (E u)
    case 1: {
      for (int j=0; j<3; j++)
        dv[j][4][i] = wJ * E * (u[0]*dXdx[j][0] + u[1]*dXdx[j][1] + u[2]*dXdx[j][2]);
      v[4][i] = 0;
    } break;
    // ---- Version 2: - v (E div(u) + u \cdot grad(E))
    case 0: {
      for (int j=0; j<3; j++) {
        dv[j][4][i] = 0;
      }
      v[4][i] = divConv; // moved computation above since Stab needs this either way
    } break;
    }
    if(CtauS > 0) {
//stab
// element measure of  inverse length squared metric from X_{K,i}X_{K,j} (contract on reference index) symmetric storage [0 1 2, 1 3 4, 2, 4, 5]
// when this is contracxed with u_i u_j it measures the length of the element in the direction of the velocity vector to create a time scale when
// inverted and square root taken.
      const CeedScalar gijd[6] = {dXdx[0][0]*dXdx[0][0]+dXdx[1][0]*dXdx[1][0]+dXdx[2][0]*dXdx[2][0],
                                  dXdx[0][0]*dXdx[0][1]+dXdx[1][0]*dXdx[1][1]+dXdx[2][0]*dXdx[2][1],
                                  dXdx[0][0]*dXdx[0][2]+dXdx[1][0]*dXdx[1][2]+dXdx[2][0]*dXdx[2][2],
                                  dXdx[0][1]*dXdx[0][1]+dXdx[1][1]*dXdx[1][1]+dXdx[2][1]*dXdx[2][1],
                                  dXdx[0][1]*dXdx[0][2]+dXdx[1][1]*dXdx[1][2]+dXdx[2][1]*dXdx[2][2],
                                  dXdx[0][2]*dXdx[0][2]+dXdx[1][2]*dXdx[1][2]+dXdx[2][2]*dXdx[2][2]};

      const CeedScalar uiujgij = ( gijd[0]*u[0]*u[0] + gijd[3]*u[1]*u[1] + gijd[5]*u[2]*u[2]
                               + 2*gijd[1]*u[0]*u[1] + 2*gijd[2]*u[0]*u[2] + 2*gijd[4]*u[1]*u[2]); 
      const CeedScalar f1   = sqrt(uiujgij); 
      const CeedScalar TauS = CtauS/f1;
      divConv *=TauS;
      dv[0][4][i] += u[0]*divConv;
      dv[1][4][i] += u[1]*divConv;
      dv[2][4][i] += u[2]*divConv;
    } // end of stabilization
  } // End Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
#endif
