Hiop.jl
========

![tests](https://github.com/exanauts/Hiop.jl/workflows/tests/badge.svg)

This is a [Julia](http://julialang.org/) [MOI](https://github.com/jump-dev/MathOptInterface.jl) interface to the [HiOp](https://github.com/LLNL/hiop) HPC nonlinear solver.

## Install dependencies

* Install HiOp and set the environment variable `JULIA_HIOP_LIBRARY_PATH` to point to your `libhiop.so` library file.

* Install the Julia package

```julia
pkg> dev https://github.com/exanauts/Hiop.jl
```

## Run tests
```julia
pkg> test Hiop
```
