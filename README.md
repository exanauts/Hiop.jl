hiop.jl
========

**hiopt.jl** is a [Julia](http://julialang.org/) interface to the [HiOp](https://github.com/LLNL/hiop) HPC nonlinear solver.

## Install dependencies

First, ensure that Hiop's shared library is in `LD_LIBRARY_PATH`, or
the environment variable `JULIA_HIOP_LIBRARY` is set.
Then, open a Julia REPL and instantiate the environment:
```julia
pkg> activate .
pkg> instantiate
```

It remains to build Hiop with:
```julia
pkg> build Hiop
```

You are now able to load Hiop in Julia:
```julia
julia> Hiop
```


## Run tests
You could run the tests either via
```julia
julia> include("test/runtests.jl")
```
or via
```julia
pkg> test Hiop
```

