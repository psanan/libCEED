# libCEED: Examples

This page provides a brief description of the examples for the libCEED library.

## Basic libCEED Examples

Two examples are provided that rely only upon libCEED without any external
libraries.

### Example 1: ex1-volume

This example uses the mass matrix to compute the length, area, or volume of a
region, depending upon runtime parameters.

### Example 2: ex2-surface

This example uses the diffusion matrix to compute the surface area of a region,
depending upon runtime parameters.

## Bakeoff Problems

This section provides a brief description of the bakeoff problems, used as examples
for the libCEED library. These bakeoff problems are high-order benchmarks designed
to test and compare the performance of high-order finite element codes.

For further documentation, readers may wish to consult the
[CEED documentation](http://ceed.exascaleproject.org/bps/) of the bakeoff problems.

### Bakeoff Problem 1

Bakeoff problem 1 is the *L<sup>2</sup>* projection problem into the finite element space.

The supplied examples solve *_B_ u = f*, where *_B_* is the mass matrix.

The nodal points, *p*, are Gauss-Legendre-Lobatto, and the quadrature points, *q* are Gauss-Legendre. There is one more quadrature point in each dimension than nodal point, *q = p + 1*.

### Bakeoff Problem 2

Bakeoff problem 2 is the *L<sup>2</sup>* projection problem into the finite element space on a vector system.

The supplied examples solve *_B_ _u_ = f*, where *_B_* is the mass matrix.

The nodal points, *p*, are Gauss-Legendre-Lobatto, and the quadrature points, *q* are Gauss-Legendre. There is one more quadrature point in each dimension than nodal point, *q = p + 1*.

### Bakeoff Problem 3

Bakeoff problem 3 is the Poisson problem.

The supplied examples solve *_A_ u = f*, where *_A_* is the Poisson operator.

The nodal points, *p*, are Gauss-Legendre-Lobatto, and the quadrature points, *q* are Gauss-Legendre. There is one more quadrature point in each dimension than nodal point, *q = p + 1*.

### Bakeoff Problem 4

Bakeoff problem 4 is the Poisson problem on a vector system.

The supplied examples solve *_A_ _u_ = f*, where *_A_* is the Laplace operator for the Poisson equation.

The nodal points, *p*, are Gauss-Legendre-Lobatto, and the quadrature points, *q* are Gauss-Legendre. There is one more quadrature point in each dimension than nodal point, *q = p + 1*.

### Bakeoff Problem 5

Bakeoff problem 5 is the Poisson problem.

The supplied examples solve *_A_ u = f*, where *_A_* is the Poisson operator.

The nodal points, *p*, are Gauss-Legendre-Lobatto, and the quadrature points, *q* are Gauss-Legendre-Lobatto. The nodal points and quadrature points are collocated.

### Bakeoff Problem 6

Bakeoff problem 6 is the Poisson problem on a vector system.

The supplied examples solve *_A_ _u_ = f*, where *_A_* is the Laplace operator for the Poisson equation.

The nodal points, *p*, are Gauss-Legendre-Lobatto, and the quadrature points, *q* are Gauss-Legendre-Lobatto. The nodal points and quadrature points are collocated.

## PETSc+libCEED Navier-Stokes Solver

The Navier-Stokes problem solves the compressible Navier-Stokes equations using an explicit time integration. A more detailed description of the problem formulation
can be found in the [`navier-stokes`](./navierstokes) folder.

## PETSc+libCEED Surface Area Examples

These examples use the mass operator to compute the surface area of a cube or a discrete cubed-sphere, using PETSc.

These examples show in particular the constructions of geometric factors to handle problems in which the elements topological dimension is different from the
geometrical dimension and for which the coordinate transformation Jacobian from the 2D reference space to a manifold embedded in 3D physical space is a non-square matrix.

## PETSc+libCEED Bakeoff Problems on the Cubed-Sphere

These examples reproduce the Bakeoff Problems 1-6 on a discrete cubed-sphere, using PETSc.

## Running Examples

To build the examples, set the `MFEM_DIR`, `PETSC_DIR` and `NEK5K_DIR` variables
and run:

```console
# libCEED examples on CPU and GPU
cd ceed
make
./ex1-volume -ceed /cpu/self
./ex1-volume -ceed /gpu/occa
./ex2-surface -ceed /cpu/self
./ex2-surface -ceed /gpu/occa
cd ..

# MFEM+libCEED examples on CPU and GPU
cd mfem
make
./bp1 -ceed /cpu/self -no-vis
./bp3 -ceed /gpu/occa -no-vis
cd ..

# Nek5000+libCEED examples on CPU and GPU
cd nek
make
./nek-examples.sh -e bp1 -ceed /cpu/self -b 3
./nek-examples.sh -e bp3 -ceed /gpu/occa -b 3
cd ..

# PETSc+libCEED examples on CPU and GPU
cd petsc
make
./bps -problem bp1 -ceed /cpu/self
./bps -problem bp2 -ceed /gpu/occa
./bps -problem bp3 -ceed /cpu/self
./bps -problem bp4 -ceed /gpu/occa
./bps -problem bp5 -ceed /cpu/self
./bps -problem bp6 -ceed /gpu/occa
cd ..

cd petsc
make
./bpsraw -problem bp1 -ceed /cpu/self
./bpsraw -problem bp2 -ceed /gpu/occa
./bpsraw -problem bp3 -ceed /cpu/self
./bpsraw -problem bp4 -ceed /gpu/occa
./bpsraw -problem bp5 -ceed /cpu/self
./bpsraw -problem bp6 -ceed /gpu/occa
cd ..

cd petsc
make
./bpssphere -problem bp1 -ceed /cpu/self
./bpssphere -problem bp2 -ceed /gpu/occa
./bpssphere -problem bp3 -ceed /cpu/self
./bpssphere -problem bp4 -ceed /gpu/occa
./bpssphere -problem bp5 -ceed /cpu/self
./bpssphere -problem bp6 -ceed /gpu/occa
cd ..

cd petsc
make
./area -problem cube -ceed /cpu/self -petscspace_degree 3
./area -problem cube -ceed /gpu/occa -petscspace_degree 3
./area -problem sphere -ceed /cpu/self -petscspace_degree 3 -dm_refine 2
./area -problem sphere -ceed /gpu/occa -petscspace_degree 3 -dm_refine 2

cd navier-stokes
make
./navierstokes -ceed /cpu/self -petscspace_degree 1
./navierstokes -ceed /gpu/occa -petscspace_degree 1
cd ..
```

The above code assumes a GPU-capable machine with the OCCA backend 
enabled. Depending on the available backends, other CEED resource specifiers can
be provided with the `-ceed` option. Other command line arguments can be found in the
[`petsc`](./petsc/README.md) folder.
