using Libdl, Base.Sys

depsfile = joinpath(dirname(@__FILE__), "deps.jl")

if isfile(depsfile)
    rm(depsfile)
end

function write_depsfile(libpath)
    open(depsfile,"w") do f
        print(f,"const libhiop = ")
        show(f, libpath)
        println(f)
    end
end

# Name of shared library
libhiop_name = string("libhiop", ".", Libdl.dlext)
paths_to_try = String[]

# First, look into JULIA_HIOP_LIBRARY_PATH
if haskey(ENV, "JULIA_HIOP_LIBRARY_PATH")
    push!(paths_to_try, joinpath(ENV["JULIA_HIOP_LIBRARY_PATH"], libhiop_name))
end
# And look also in LD_LIBRARY_PATH
push!(paths_to_try, libhiop_name)

global found_hiop = false
for path in paths_to_try
    println(path)
    d = Libdl.dlopen_e(path)
    if d != C_NULL
        global found_hiop = true
        # Store pointer to Hiop
        write_depsfile(path)
        break
    end
end

if !found_hiop
    error("Could not load HiOp shared library. ",
            "Make sure it is in your LD_LIBRARY_PATH.")
end
