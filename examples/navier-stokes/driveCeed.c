#include <ceed.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "assert.h"
#include "common.h"
#include "advection.h"
   
 int main(int argc, char **argv) {

//int    driveceed(double* y,   double* ac,  
//     	double* x,         int* ien,   
//     	double* rest)     
//{
//
//  Can test by replacing passed data above by data written from PHASTA into the file data4libCEED.dat 
// written as follows (fortran)
//      open(unit=777, file='data4libCEED.dat',status='unknown')
//      write(777,*) npro,nen 
//      do i =1,npro
//        write(777,*) (ien(i,j), j=1,nen)
//      enddo
//      write(777,*) numnp,nsd
//      do i=1,numnp
//        write(777,*) (point2x(i,j),j=1,nsd)
//      enddo
//      write(777,*) numnp,ndof
//      do i=1,numnp
//         write(777,*) (y(i,j),j=1,ndof)
//      enddo
//      write(777,*) numnp,ndof
//      do i=1,numnp
//         write(777,*) (ac(i,j),j=1,ndof)
//      enddo
//      close(777)
//   note, rest is output data thus not needed
// numbers comming from common.h (PHASTA's not libCEEDs)   
// nshg=numnp
// nshl=nen (for this case)


// 
// ----------------------------------------------------------------------
// 
//  This is the libCEED driver routine.
// 
// input:
//  y      (nshg,ndof)           : Y-variables at n+alpha_v
//  ac     (nshg,ndof)           : Primvar. accel. variable n+alpha_m
//  acold  (nshg,ndof)           : Primvar. accel. variable at begng step
//  x      (numnp,nsd)           : node coordinates
//  ien    (npro,nshl)           : connectivity
// 
// output:
//  rest    (nshg)           : residual
// 
//  
// Jansen 2019
// ----------------------------------------------------------------------
// 


// Get variables from common_c.h
      int tmp32,nen,numnp,nshg, nflow, nsd, iownnodes;
//      nshg  = conpar.nshg; 
//      nflow = conpar.nflow; 
//      numnp = conpar.numnp; 
//      nen = conpar.nen; 
      int node, element, var, eqn;
      double valtoinsert;
      int nenl, iel, lelCat, lcsyst, iorder;
      int mattyp, ndofl, nsymdl, npro, ngauss, nppro;
//      npro = propar.npro; 
// DEBUG
      int i,j,k,l,m;

      // FIXME: PetscScalar
      double  real_rtol, real_abstol, real_dtol;
// /DEBUG
//
//  Can test by replacing passed data above by data written from PHASTA into the file data4libCEED.dat 
// written as follows (fortran)
//      open(unit=777, file='data4libCEED.dat',status='unknown')
//      write(777,*) npro,nen 
//      do i =1,npro
//        write(777,*) (ien(i,j), j=1,nen)
//      enddo
//      write(777,*) numnp,nsd
//      do i=1,numnp
//        write(777,*) (point2x(i,j),j=1,nsd)
//      enddo
//      write(777,*) numnp,ndof
//      do i=1,numnp
//         write(777,*) (y(i,j),j=1,ndof)
//      enddo
//      write(777,*) numnp,ndof
//      do i=1,numnp
//         write(777,*) (ac(i,j),j=1,ndof)
//      enddo
//      close(777)
//   note, rest is output data thus not needed
// numbers comming from common.h (PHASTA's not libCEEDs)   
// nshg=numnp
// nshl=nen (for this case)
//      open(unit=777, file='data4libCEED.dat',status='unknown')
//        FILE *fopen(const char *filename, const char *mode);
        FILE *fp; 
        double myvariable;
        fp=fopen("data4libCEED.dat", "r");
// connectivity
        fscanf(fp,"%d",&npro);
        fscanf(fp,"%d",&nen);
        int ien[npro*nen];
        for(int i = 0; i < npro; i++) {
    	  for (int j = 0 ; j < nen; j++) {
            fscanf(fp,"%d",&ien[i+j*npro]);
          }
        }
// coordinates
        fscanf(fp,"%d",&numnp);
        fscanf(fp,"%d",&nsd);
        double x[numnp*nsd];
        for(int i = 0; i < numnp; i++) {
    	  for (int j = 0 ; j < nsd; j++) {
            fscanf(fp,"%lf",&x[i+j*numnp]);
          }
        }
        nshg=numnp;
        double y[numnp*nsd];
        double ac[numnp*nsd];
        double rest[numnp*nsd];

      double qfp[5*numnp]; // libCeed will get this array with scalar thrown into the temperature slot since advection works that way later we will need to make a real scalar equation. 
      double qdotfp[5*numnp];
      double q0[5*numnp]; // just for debugging using ics
      double x0[3*numnp]; // just for debugging using ics
        
// solution not needed yet
//        fscanf(fp,"%d",&numnp);
//        fscanf(fp,"%d",&nsd);
//        double x[numnp*nsd];
//        for(int i = 0; i < numnp; i++) {
//    	  for (int j = 0 ; j < nsd; j++) {
//            fscanf(fp,"%lf",&x[i+j*nsd]);
////             printf("%.15f ",myvariable);
//          }
////          printf("\n");
//        }
//      write(777,*) npro,nen 
//      do i =1,npro
//        write(777,*) (ien(i,j), j=1,nen)
//      enddo
//      write(777,*) numnp,nsd
//      do i=1,numnp
//        write(777,*) (point2x(i,j),j=1,nsd)
//      enddo
//      write(777,*) numnp,ndof
//      do i=1,numnp
//         write(777,*) (y(i,j),j=1,ndof)
//      enddo
//      write(777,*) numnp,ndof
//      do i=1,numnp
//         write(777,*) (ac(i,j),j=1,ndof)
//      enddo
//      close(777)
//   note, rest is output data thus not needed
// numbers comming from common.h (PHASTA's not libCEEDs)   
// nshg=numnp
// nshl=nen (for this case)

  Ceed ceed;
  CeedElemRestriction restrictx, restrictq, restrictxi, restrictqdi,restrictxFake, restrictxcoord;
  CeedBasis basisxc, bx, bq; 
  CeedQFunction qf_setup, qf_ifunction, qf_ics;
  CeedOperator op_setup, op_ifunction,  op_ics;
  CeedVector qdata, X, U, Udot, V, Xfake;
  CeedVector xceed, q0ceed, qceed, qdotceed, gceed;
  const CeedScalar *hv;
  const CeedScalar *hvt;
  CeedInt nelem = npro, P = 2, Q = 2, qpownsd=Q*Q*Q;
  CeedInt nshl=P*P*P, qdatasize=10;
  CeedInt indx[npro*nen], indq[npro*nshl];
  CeedScalar xref[24];
  CeedScalar theta0     = 300.;     // K
  CeedScalar thetaC     = -15.;     // K
  CeedScalar P0         = 1.e5;     // Pa
  CeedScalar N          = 0.01;     // 1/s
  CeedScalar cv         = 717.;     // J/(kg K)
  CeedScalar cp         = 1004.;    // J/(kg K)
  CeedScalar g          = 9.81;     // m/s^2
  CeedScalar lx        = 8000.;    // m
  CeedScalar ly        = 8000.;    // m
  CeedScalar lz        = 4000.;    // m
  CeedScalar rc         = 1000.;    // m (Radius of bubble)
  CeedScalar Rd;
  CeedInt periodicity[3];
  Rd=8.314; //PHASTA VALUE cp-cv;

  CeedScalar ctxSetup[] = {theta0, thetaC, P0, N, cv, cp, Rd, g, rc,
                           lx, ly, lz,
                           periodicity[0], periodicity[1], periodicity[2],
                          };


//  ! [Ceed Init]
//  const char* intStr="/cpu/self/ref/memcheck";
  const char* intStr="/cpu/self/ref/serial";
  CeedInit(intStr, &ceed);
//! [Ceed Init]
  for (CeedInt i=0; i<nelem; i++) {
    for (CeedInt j=0; j<nen; j++) 
    if(1) { // this numbering make GL find x,yz, with first 8 the local nodes of the first element
      indx[j+i*nen] = ien[i+j*nelem]-1; // transpose and shift to C-based node numbering
    }  else { 
      indx[j+i*nen] = ien[j+i*nen]-1; // transpose and shift to C-based node numbering
    }
  }
  for (CeedInt i=0; i<nelem; i++) {
   if(1) {   // swap also has to change if you don't transpose idx
      tmp32=indx[3+i*nen];
      indx[3+i*nen]=indx[2+i*nen];
      indx[2+i*nen]=tmp32;
      tmp32=indx[7+i*nen];
      indx[7+i*nen]=indx[6+i*nen];
      indx[6+i*nen]=tmp32;
    } else {
      tmp32=indx[3*nelem+i];
      indx[3*nelem+i]=indx[2*nelem+i];
      indx[2*nelem+i]=tmp32;
      tmp32=indx[7*nelem+i];
      indx[7*nelem+i]=indx[6*nelem+i];
      indx[6*nelem+i]=tmp32;
   }
  }
// jiggle the coordinates with a shift scaling off their node number to "encode" node number 
  for (CeedInt j=0; j<numnp; j++) {
      x[j]+=0.000001*j;
      x[j+numnp]+=0.000001*j;
      x[j+numnp*2]+=0.000001*j;
   }
  CeedVectorCreate(ceed, numnp*3, &X); //ns757
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x); // the jiggled  coordinates will go INTO IC and solution

//! [Basis Create]
  CeedBasisCreateTensorH1Lagrange(ceed, 3, 3, 2, Q, CEED_GAUSS, &bx);
  CeedBasisCreateTensorH1Lagrange(ceed, 3, 5, P, Q, CEED_GAUSS, &bq);

//! [ElemRestr Create]
  CeedElemRestrictionCreate(ceed, npro, nen, numnp, 3, CEED_MEM_HOST, CEED_USE_POINTER, indx, &restrictx); // coordinates
  CeedElemRestrictionCreate(ceed, npro, nshl, nshg, 5, CEED_MEM_HOST, CEED_USE_POINTER, indx, &restrictq); // solution: change to indq if HO
  CeedElemRestrictionCreateIdentity(ceed, npro, qpownsd, qpownsd*npro, qdatasize, &restrictqdi); //metrics shared from setup to residual
  CeedElemRestrictionCreateIdentity(ceed, npro, qpownsd, qpownsd*npro, 1, &restrictxi); // weight
//CeedElemRestrictionCreateIdentity(ceed, localNelem, numQdim,
//                                                         localNelem*numQdim, 1,
//                                                                        &restrictxi); //K G_xi    3 lines from nsplex.c

//! [QFunction Create]
  CeedQFunctionCreateInterior(ceed, 1, Setup, Setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "dx", nsd*nsd, CEED_EVAL_GRAD);
  if (1) { // for debugging I removed weight from argument list to prove that it was the issue...then it became clear that it was my swapped order that corrupted
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
//CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT); line 782 in nsplex
  }
  CeedQFunctionAddOutput(qf_setup, "qdata", qdatasize, CEED_EVAL_NONE);
  // Create the operator that builds the quadrature data for the NS operator
  CeedOperatorCreate(ceed, qf_setup, NULL, NULL, &op_setup);
  CeedOperatorSetField(op_setup, "dx", restrictx, CEED_NOTRANSPOSE, //K AddInput says gradient using basisx of operatorApply defined from xcorners
                       bx, CEED_VECTOR_ACTIVE);
  if (1) { // other part of removing it. 
  CeedOperatorSetField(op_setup, "weight", restrictxi, CEED_NOTRANSPOSE, //K get the weight to setup Q function 
                       bx, CEED_VECTOR_NONE);
//CeedOperatorSetField(op_setup, "weight", restrictxi, CEED_NOTRANSPOSE, //K get the weight to setup Q function 
//                 basisx, CEED_VECTOR_NONE);  // 2 lines from nsplex.c

  }
  CeedOperatorSetField(op_setup, "qdata", restrictqdi, CEED_NOTRANSPOSE,//K output of setup Q function metric data at quadrature points to share with op
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedVectorCreate(ceed, qdatasize*npro*qpownsd, &qdata);

  CeedOperatorApply(op_setup, X, qdata, CEED_REQUEST_IMMEDIATE); //K Apply setup for the selected case.  Creates qdata: WdetJ and dxidx at qpts

if(1) { //HACK TEST  to use IC to look at GL view of coordinates
   CeedVectorCreate(ceed, 3*nshg, &xceed); // this is the vector RETURNED from ICS  potenitally higher order NODE locations
   for (CeedInt i=0; i< 3*nshg; i++) x0[i]=0.0;
   CeedVectorSetArray(xceed, CEED_MEM_HOST, CEED_USE_POINTER, x0);
//  If you do this do you still have to set array  
//  CeedElemRestrictionCreateVector(restrictq, &q0ceed, NULL); // jns745
//  CeedElemRestrictionCreateVector(restrictxcoord, &xceed, NULL); //K passed out of ics (not used)
// so just do the direct way below.  Unclear if we realy have to initialize to zero but being safe
   
   CeedVectorCreate(ceed, 5*numnp, &q0ceed); // ns 766
   for (CeedInt i=0; i< 5*numnp; i++) q0[i]=0.0;
   CeedVectorSetArray(q0ceed, CEED_MEM_HOST, CEED_USE_POINTER, q0);

  CeedBasisCreateTensorH1Lagrange(ceed, 3, 3, 2, Q, CEED_GAUSS_LOBATTO, &basisxc); 
  CeedQFunctionCreateInterior(ceed, 1, ICsAdvection, ICsAdvection_loc, &qf_ics);
  if(1) { // a 0 sends gradient to ICS. needs a same switch in ICS to receive J . 
    CeedQFunctionAddInput(qf_ics, "x", 3, CEED_EVAL_INTERP); //K comments on this are in SetField Below
  } else {
    CeedQFunctionAddInput(qf_ics, "x", nsd*nsd, CEED_EVAL_GRAD); //K comments on this are in SetField Below
  }
  CeedQFunctionAddOutput(qf_ics, "q0", 5, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_ics, "coords", 3, CEED_EVAL_NONE);
  CeedQFunctionSetContext(qf_ics, &ctxSetup, sizeof ctxSetup);
  CeedOperatorCreate(ceed, qf_ics, NULL, NULL, &op_ics);
  if(1) { // change to zero to get coordinates of the quadrature points instead of element nodes (also for metrics at q pt if changed above)
    CeedOperatorSetField(op_ics, "x", restrictx, CEED_NOTRANSPOSE, basisxc, CEED_VECTOR_ACTIVE);
  } else { 
    CeedOperatorSetField(op_ics, "x", restrictx, CEED_NOTRANSPOSE, bx, CEED_VECTOR_ACTIVE);
  }
  CeedOperatorSetField(op_ics, "q0", restrictq, CEED_TRANSPOSE, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
//  CeedElemRestrictionCreateIdentity(ceed, npro, qpownsd, qpownsd*npro, nsd, &restrictxcoord);  WRONG  they assemble nodes on top of each other back to an Lvector....just use restricxtx   FOR THIS CASE wher P=2 for borh solution AND coorinates
  CeedOperatorSetField(op_ics, "coords", restrictx, CEED_NOTRANSPOSE, CEED_BASIS_COLLOCATED, xceed);

  CeedOperatorApply(op_ics, X, q0ceed, CEED_REQUEST_IMMEDIATE); // jns889 
//CeedVectorGetArrayRead(q0ceed, CEED_MEM_HOST, &hvt);
//  for (CeedInt i=0; i<numnp*3; i++)
//    qTest[i]=hvt[i];
//  CeedVectorRestoreArrayRead(q0ceed, &hvt);
}
//END HACK TEST

// The ifunction I have not made it too yet because qdata is garbage
  CeedQFunctionCreateInterior(ceed, 1, IFunction_Advection, IFunction_Advection_loc, &qf_ifunction);
  CeedQFunctionAddInput(qf_ifunction, "qdata", qdatasize, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_ifunction, "qdot", 5, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_ifunction, "q", 5, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_ifunction, "dq", 5*nsd, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_ifunction, "v", 5, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_ifunction, "dv", 5*nsd, CEED_EVAL_GRAD);


  CeedVectorCreate(ceed, 5*nshg, &U);
  for (CeedInt i=0; i< nshg; i++) {
    qfp[i]=y[i]/Rd/y[i+4*nshg]; // density
    qfp[i+  nshg]=qfp[i]*y[i+1*nshg]; // density*u1
    qfp[i+2*nshg]=qfp[i]*y[i+2*nshg]; // density*u2
    qfp[i+3*nshg]=qfp[i]*y[i+3*nshg]; // density*u3
    qfp[i+4*nshg]=     y[i+5*nshg]; // PHASTA scalar
    qdotfp[i+4*nshg]= ac[i+5*nshg]; // PHASTA scalar
  }
  CeedVectorSetArray(U, CEED_MEM_HOST, CEED_USE_POINTER, qfp);

  CeedVectorCreate(ceed, 5*nshg, &Udot);
  CeedVectorSetArray(Udot, CEED_MEM_HOST, CEED_USE_POINTER, qdotfp);
  CeedVectorCreate(ceed, 5*nshg, &V);


  { // Create the IFunction operator  
    CeedOperatorCreate(ceed, qf_ifunction, NULL, NULL, &op_ifunction);
    CeedOperatorSetField(op_ifunction, "q", restrictq, CEED_NOTRANSPOSE, //K Active input is current solution vector Q set on OperatorApply line q=B_q_i G_q Q  
                         bq, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_ifunction, "dq", restrictq, CEED_TRANSPOSE, //K Active input is current solution vector Q set on OperatorApply line q=B_q_{gi} G_q Q  
                         bq, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_ifunction, "qdot", restrictq, CEED_NOTRANSPOSE, //K not an active vector but the is qdot (like ac in PHASTA)
                         bq, Udot);
    CeedOperatorSetField(op_ifunction, "qdata", restrictqdi, CEED_NOTRANSPOSE, //K shared data from setup is "set"  
                         CEED_BASIS_COLLOCATED, qdata);
    CeedOperatorSetField(op_ifunction, "v", restrictq, CEED_TRANSPOSE, //K Output 
                         bq, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_ifunction, "dv", restrictq, CEED_TRANSPOSE, //K Output
                         bq, CEED_VECTOR_ACTIVE);
  }
  double CtauS=0.5;
  int strong_form=0;
  int stab=2;
  struct Advection2dContext_ ctxAdvection2d = { //K struct that passes data needed at quadrature points for both advection
    .CtauS = CtauS,
    .strong_form = strong_form,
    .stabilization = stab,
  };
  CeedQFunctionSetContext(qf_ifunction, &ctxAdvection2d, sizeof ctxAdvection2d); //K This function associates the struct with qf_rhs and in next line qf_ifunction
// Calculate qdata 
  // Apply Setup Ceed Operators

  CeedOperatorApply(op_ifunction, U, V, CEED_REQUEST_IMMEDIATE); //K Apply setup for the selected case.  Creates qdata: WdetJ and dxidx at qpts

  CeedVectorGetArrayRead(V, CEED_MEM_HOST, &hv);
  for (CeedInt i=0; i<nshg; i++)
    rest[i]=hv[i];
  CeedVectorRestoreArrayRead(V, &hv);

// pausing here to line 897 of copying from nsplex.c

//  I know I  have not finished the cleanup below
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_ifunction);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_ifunction);
  CeedElemRestrictionDestroy(&restrictq);
  CeedElemRestrictionDestroy(&restrictx);
  CeedElemRestrictionDestroy(&restrictxi);
  CeedElemRestrictionDestroy(&restrictqdi);
  CeedBasisDestroy(&bq);
  CeedBasisDestroy(&bx);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&Udot);
  CeedVectorDestroy(&V);
  CeedVectorDestroy(&qdata);
  CeedDestroy(&ceed);
  return 0;
// .... end
}
