Hiop.jl
========

![tests](https://github.com/exanauts/Hiop.jl/workflows/tests/badge.svg)

This is a [Julia](http://julialang.org/) [MOI](https://github.com/jump-dev/MathOptInterface.jl) interface to the [HiOp](https://github.com/LLNL/hiop) HPC nonlinear solver.

## Install dependencies

First, ensure that Hiop's shared library is in `LD_LIBRARY_PATH`, or
the environment variable `JULIA_HIOP_LIBRARY_PATH` is set.
Then, open a Julia REPL and instantiate the environment:
```julia
pkg> dev https://github.com/exanauts/Hiop.jl
```

It remains to build Hiop with:
```julia
pkg> build Hiop
```

You are now able to load Hiop in Julia:
```julia
julia> using Hiop
```


## Run tests
```julia
pkg> test Hiop
```

## Using dense or sparse algebra in HiOp

```julia
# Dense algebra
model = Model(optimizer_with_attributes(Hiop.Optimizer, "algebra" => :Dense))
# Sparse algebra
model = Model(optimizer_with_attributes(Hiop.Optimizer, "algebra" => :Sparse))
```
