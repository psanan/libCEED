# Getting Started

## Building

The CEED library, `libceed`, is a C99 library with no external dependencies. The library
has Fortran and Python interfaces; see `interface/ceed-fortran.c` and
`interface/ceed-python/`. It can be built using

    make

or, with optimization flags

    make OPT='-O3 -march=skylake-avx512 -ffp-contract=fast'

These optimization flags are used by all languages (C, C++, Fortran) and this
makefile variable can also be set for testing and examples (below).

The library attempts to automatically detect support for the AVX
instruction set using gcc-style compiler options for the host.
Support may need to be manually specified via

    make AVX=1

or

    make AVX=0

if your compiler does not support gcc-style options, if you are cross
compiling, etc.

## Testing

The test suite produces [TAP](https://testanything.org) output and is run by:

    make test

or, using the `prove` tool distributed with Perl (recommended)

    make prove

## Backends

There are multiple supported backends, which can be selected at runtime in the examples:

| CEED resource            | Backend                                           |
| :----------------------- | :------------------------------------------------ |
| `/cpu/self/ref/serial`   | Serial reference implementation                   |
| `/cpu/self/ref/blocked`  | Blocked refrence implementation                   |
| `/cpu/self/memcheck`     | Memcheck backend, undefined value checks          |
| `/cpu/self/opt/serial`   | Serial optimized C implementation                 |
| `/cpu/self/opt/blocked`  | Blocked optimized C implementation                |
| `/cpu/self/avx/serial`   | Serial AVX implementation                         |
| `/cpu/self/avx/blocked`  | Blocked AVX implementation                        |
| `/cpu/self/xsmm/serial`  | Serial LIBXSMM implementation                     |
| `/cpu/self/xsmm/blocked` | Blocked LIBXSMM implementation                    |
| `/cpu/occa`              | Serial OCCA kernels                               |
| `/gpu/occa`              | CUDA OCCA kernels                                 |
| `/omp/occa`              | OpenMP OCCA kernels                               |
| `/ocl/occa`              | OpenCL OCCA kernels                               |
| `/gpu/cuda/ref`          | Reference pure CUDA kernels                       |
| `/gpu/cuda/reg`          | Pure CUDA kernels using one thread per element    |
| `/gpu/cuda/shared`       | Optimized pure CUDA kernels using shared memory   |
| `/gpu/cuda/gen`          | Optimized pure CUDA kernels using code generation |
| `/gpu/magma`             | CUDA MAGMA kernels                                |

The `/cpu/self/*/serial` backends process one element at a time and are intended for meshes
with a smaller number of high order elements. The `/cpu/self/*/blocked` backends process
blocked batches of eight interlaced elements and are intended for meshes with higher numbers
of elements.

The `/cpu/self/ref/*` backends are written in pure C and provide basic functionality.

The `/cpu/self/opt/*` backends are written in pure C and use partial e-vectors to improve performance.

The `/cpu/self/avx/*` backends rely upon AVX instructions to provide vectorized CPU performance.

The `/cpu/self/xsmm/*` backends rely upon the [LIBXSMM](http://github.com/hfp/libxsmm) package
to provide vectorized CPU performance. If linking MKL and LIBXSMM is desired but
the Makefile is not detecting `MKLROOT`, linking libCEED against MKL can be
forced by setting the environment variable `MKL=1`.

The `/cpu/self/memcheck/*` backends rely upon the [Valgrind](http://valgrind.org/) Memcheck tool
to help verify that user QFunctions have no undefined values. To use, run your code with
Valgrind and the Memcheck backends, e.g. `valgrind ./build/ex1 -ceed /cpu/self/ref/memcheck`. A
'development' or 'debugging' version of Valgrind with headers is required to use this backend.
This backend can be run in serial or blocked mode and defaults to running in the serial mode
if `/cpu/self/memcheck` is selected at runtime.

The `/*/occa` backends rely upon the [OCCA](http://github.com/libocca/occa) package to provide
cross platform performance.

The `/gpu/cuda/*` backends provide GPU performance strictly using CUDA.

The `/gpu/magma` backend relies upon the [MAGMA](https://bitbucket.org/icl/magma) package.

## Examples

libCEED comes with several examples of its usage, ranging from standalone C
codes in the `/examples/ceed` directory to examples based on external packages,
such as MFEM, PETSc, and Nek5000. Nek5000 v18.0 or greater is required.

To build the examples, set the `MFEM_DIR`, `PETSC_DIR` and `NEK5K_DIR` variables
and run:

```console
# libCEED examples on CPU and GPU
cd examples/ceed
make
./ex1-volume -ceed /cpu/self
./ex1-volume -ceed /gpu/occa
./ex2-surface -ceed /cpu/self
./ex2-surface -ceed /gpu/occa
cd ../..

# MFEM+libCEED examples on CPU and GPU
cd examples/mfem
make
./bp1 -ceed /cpu/self -no-vis
./bp3 -ceed /gpu/occa -no-vis
cd ../..

# Nek5000+libCEED examples on CPU and GPU
cd examples/nek
make
./nek-examples.sh -e bp1 -ceed /cpu/self -b 3
./nek-examples.sh -e bp3 -ceed /gpu/occa -b 3
cd ../..

# PETSc+libCEED examples on CPU and GPU
cd examples/petsc
make
./bps -problem bp1 -ceed /cpu/self
./bps -problem bp2 -ceed /gpu/occa
./bps -problem bp3 -ceed /cpu/self
./bps -problem bp4 -ceed /gpu/occa
./bps -problem bp5 -ceed /cpu/self
./bps -problem bp6 -ceed /gpu/occa
cd ../..

cd examples/petsc
./area -problem cube -ceed /cpu/self -petscspace_degree 3
./area -problem cube -ceed /gpu/occa -petscspace_degree 3
./area -problem sphere -ceed /cpu/self -petscspace_degree 3 -dm_refine 2
./area -problem sphere -ceed /gpu/occa -petscspace_degree 3 -dm_refine 2
cd ../..

cd examples/navier-stokes
make
./navierstokes -ceed /cpu/self
./navierstokes -ceed /gpu/occa
cd ../..
```

The above code assumes a GPU-capable machine with the OCCA backend
enabled. Depending on the available backends, other Ceed resource specifiers can
be provided with the `-ceed` option.

## Benchmarks

A sequence of benchmarks for all enabled backends can be run using

```console
make benchmarks
```

The results from the benchmarks are stored inside the `benchmarks/` directory
and they can be viewed using the commands (requires python with matplotlib):

```console
cd benchmarks
python postprocess-plot.py petsc-bps-bp1-*-output.txt
python postprocess-plot.py petsc-bps-bp3-*-output.txt
```

Using the `benchmarks` target runs a comprehensive set of benchmarks which may
take some time to run. Subsets of the benchmarks can be run using the scripts in the `benchmarks` folder.

For more details about the benchmarks, see the `benchmarks/README.md` file.


## Install

To install libCEED, run

    make install prefix=/usr/local

or (e.g., if creating packages),

    make install prefix=/usr DESTDIR=/packaging/path

Note that along with the library, libCEED installs kernel sources, e.g. OCCA
kernels are installed in `$prefix/lib/okl`. This allows the OCCA backend to
build specialized kernels at run-time. In a normal setting, the kernel sources
will be found automatically (relative to the library file `libceed.so`).
However, if that fails (e.g. if `libceed.so` is moved), one can copy (cache) the
kernel sources inside the user OCCA directory, `~/.occa` using

    $(OCCA_DIR)/bin/occa cache ceed $(CEED_DIR)/lib/okl/*.okl

This will allow OCCA to find the sources regardless of the location of the CEED
library. One may occasionally need to clear the OCCA cache, which can be accomplished
by removing the `~/.occa` directory or by calling `$(OCCA_DIR)/bin/occa clear -a`.

To install libCEED for Python, run

    python setup.py build install

with the desired setuptools options, such as `--user`.

Alternatively, if libCEED is installed in the directory specified by the
environment variable `CEED_DIR`, then run

    pip install .

### pkg-config

In addition to library and header, libCEED provides a [pkg-config][pkg-config1]
file that can be used to easily compile and link. [For example][pkg-config2], if
`$prefix` is a standard location or you set the environment variable
`PKG_CONFIG_PATH`,

    cc `pkg-config --cflags --libs ceed` -o myapp myapp.c

will build `myapp` with libCEED.  This can be used with the source or
installed directories.  Most build systems have support for pkg-config.

## Contact

You can reach the libCEED team by emailing [ceed-users@llnl.gov](mailto:ceed-users@llnl.gov)
or by leaving a comment in the [issue tracker](https://github.com/CEED/libCEED/issues).

## Copyright

The following copyright applies to each file in the CEED software suite, unless
otherwise stated in the file:

> Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the
> Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights reserved.

See files LICENSE and NOTICE for details.

[ceed-soft]:   http://ceed.exascaleproject.org/software/
[ecp]:         https://exascaleproject.org/exascale-computing-project
[pkg-config1]: https://en.wikipedia.org/wiki/Pkg-config
[pkg-config2]: https://people.freedesktop.org/~dbn/pkg-config-guide.html#faq
