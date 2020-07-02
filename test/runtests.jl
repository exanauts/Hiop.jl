using Test

@testset "C API" begin
  global ns = 400
  global nd = 100
  include(joinpath(dirname(@__FILE__), "..", "example", "ex4.jl"))
  @test nlp.obj_val ≈ -4.999509728895e+01
end

@testset "MathOptInterface" begin
    include("MOI_wrapper.jl")
end
