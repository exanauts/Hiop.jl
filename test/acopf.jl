using JuMP
using Ipopt
using Hiop
using DelimitedFiles
using LinearAlgebra

include("opfdata.jl")

# export solve, model, initialPt_IPOPT, outputAll, computeAdmitances
function solve(opfmodel, opf_data)
    optimize!(opfmodel)
    status = termination_status(opfmodel)
    if status != MOI.LOCALLY_SOLVED
        println("Could not solve the model to optimality.")
    end
    return opfmodel,status
end

function model(opf_data; max_iter=100, solver="Ipopt")
    Pg0, Qg0, Vm0, Va0 = initialPt_IPOPT(opf_data)
    lines = opf_data.lines; buses = opf_data.buses; generators = opf_data.generators; baseMVA = opf_data.baseMVA
    busIdx = opf_data.BusIdx; FromLines = opf_data.FromLines; ToLines = opf_data.ToLines; BusGeners = opf_data.BusGenerators;

    nbus  = length(buses); nline = length(lines); ngen  = length(generators)

    #branch admitances
    YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI = computeAdmitances(lines, buses, baseMVA)

    #
    # JuMP model now
    #
    if solver == "Hiop"
        opfmodel = Model(optimizer_with_attributes(Hiop.Optimizer))
    else
        opfmodel = Model(optimizer_with_attributes(Ipopt.Optimizer, "max_iter" => max_iter))
    end

    @variable(opfmodel, generators[i].Pmin <= Pg[i=1:ngen] <= generators[i].Pmax, start = Pg0[i])
    @variable(opfmodel, generators[i].Qmin <= Qg[i=1:ngen] <= generators[i].Qmax, start = Qg0[i])

    @variable(opfmodel, Va[i=1:nbus], start = Va0[i])
    @variable(opfmodel, buses[i].Vmin <= Vm[i=1:nbus] <= buses[i].Vmax, start = Vm0[i])
    # fix the voltage angle at the reference bus
    set_lower_bound(Va[opf_data.bus_ref], buses[opf_data.bus_ref].Va)
    set_upper_bound(Va[opf_data.bus_ref], buses[opf_data.bus_ref].Va)

    # minimize active power
    coeff0 = Vector{Float64}(undef, ngen)
    for (i,v) in enumerate(generators)
        coeff0[i] = v.coeff[v.n]
    end
    coeff1 = Vector{Float64}(undef, ngen)
    for (i,v) in enumerate(generators)
        coeff1[i] = v.coeff[v.n - 1]
    end
    coeff2 = Vector{Float64}(undef, ngen)
    for (i,v) in enumerate(generators)
        coeff2[i] = v.coeff[v.n - 2]
    end

    @NLobjective(opfmodel, Min, sum( coeff2[i]*(baseMVA*Pg[i])^2 
                                    + coeff1[i]*(baseMVA*Pg[i])
                                    + coeff0[i] for i=1:ngen))

    #
    # power flow balance
    #

    for b in 1:nbus
        # real part
        @NLconstraint(
        opfmodel, 
        ( sum( YffR[l] for l in FromLines[b]) + sum( YttR[l] for l in ToLines[b]) + YshR[b] ) * Vm[b]^2 
        + sum( Vm[b]*Vm[busIdx[lines[l].to]]  *( YftR[l]*cos(Va[b]-Va[busIdx[lines[l].to]]  ) + YftI[l]*sin(Va[b]-Va[busIdx[lines[l].to]]  )) for l in FromLines[b] )  
        + sum( Vm[b]*Vm[busIdx[lines[l].from]]*( YtfR[l]*cos(Va[b]-Va[busIdx[lines[l].from]]) + YtfI[l]*sin(Va[b]-Va[busIdx[lines[l].from]])) for l in ToLines[b]   ) 
        - ( sum(baseMVA*Pg[g] for g in BusGeners[b]) - buses[b].Pd ) / baseMVA      # Sbus part
        ==0)
        # imaginary part
    end
    for b in 1:nbus
        @NLconstraint(
        opfmodel,
        ( sum(-YffI[l] for l in FromLines[b]) + sum(-YttI[l] for l in ToLines[b]) - YshI[b] ) * Vm[b]^2 
        + sum( Vm[b]*Vm[busIdx[lines[l].to]]  *(-YftI[l]*cos(Va[b]-Va[busIdx[lines[l].to]]  ) + YftR[l]*sin(Va[b]-Va[busIdx[lines[l].to]]  )) for l in FromLines[b] )
        + sum( Vm[b]*Vm[busIdx[lines[l].from]]*(-YtfI[l]*cos(Va[b]-Va[busIdx[lines[l].from]]) + YtfR[l]*sin(Va[b]-Va[busIdx[lines[l].from]])) for l in ToLines[b]   )
        - ( sum(baseMVA*Qg[g] for g in BusGeners[b]) - buses[b].Qd ) / baseMVA      #Sbus part
        ==0)
    end
    #
    # branch/lines flow limits
    #
    nlinelim=0
    for l in 1:nline
        if lines[l].rateA!=0 && lines[l].rateA<1.0e10
        nlinelim += 1
        flowmax=(lines[l].rateA/baseMVA)^2

        #branch apparent power limits (from bus)
        Yff_abs2=YffR[l]^2+YffI[l]^2;        Yft_abs2=YftR[l]^2+YftI[l]^2
        Yre=YffR[l]*YftR[l]+YffI[l]*YftI[l]; Yim=-YffR[l]*YftI[l]+YffI[l]*YftR[l]
        @NLconstraint(
            opfmodel,
            Vm[busIdx[lines[l].from]]^2 *
            ( Yff_abs2*Vm[busIdx[lines[l].from]]^2 + Yft_abs2*Vm[busIdx[lines[l].to]]^2 
            + 2*Vm[busIdx[lines[l].from]]*Vm[busIdx[lines[l].to]]
                *(Yre*cos(Va[busIdx[lines[l].from]]-Va[busIdx[lines[l].to]])
                -Yim*sin(Va[busIdx[lines[l].from]]-Va[busIdx[lines[l].to]])
                ) 
            ) 
            - flowmax <=0)

        #branch apparent power limits (to bus)
        Ytf_abs2=YtfR[l]^2+YtfI[l]^2; Ytt_abs2=YttR[l]^2+YttI[l]^2
        Yre=YtfR[l]*YttR[l]+YtfI[l]*YttI[l]; Yim=-YtfR[l]*YttI[l]+YtfI[l]*YttR[l]
        @NLconstraint(
            opfmodel,
            Vm[busIdx[lines[l].to]]^2 *
            ( Ytf_abs2*Vm[busIdx[lines[l].from]]^2 + Ytt_abs2*Vm[busIdx[lines[l].to]]^2
            + 2*Vm[busIdx[lines[l].from]]*Vm[busIdx[lines[l].to]]
                *(Yre*cos(Va[busIdx[lines[l].from]]-Va[busIdx[lines[l].to]])
                -Yim*sin(Va[busIdx[lines[l].from]]-Va[busIdx[lines[l].to]])
                )
            )
            - flowmax <=0)
        end
    end

    println("Buses: $nbus  Lines: $nline  Generators: $ngen", nbus, nline, ngen)
    println("Lines with limits: $nlinelim")

    return opfmodel, Pg, Qg, Va, Vm
end
# Compute initial point for IPOPT based on the values provided in the case data
function initialPt_IPOPT(opfdata)
    Pg=zeros(length(opfdata.generators)); Qg=zeros(length(opfdata.generators)); i=1
    for g in opfdata.generators
        # set the power levels in in between the bounds as suggested by matpower 
        # (case data also contains initial values in .Pg and .Qg - not used with IPOPT)
        Pg[i]=0.5*(g.Pmax+g.Pmin)
        Qg[i]=0.5*(g.Qmax+g.Qmin)
        i=i+1
    end
    @assert i-1==length(opfdata.generators)

    Vm=zeros(length(opfdata.buses)); i=1;
    for b in opfdata.buses
        # set the ini val for voltage magnitude in between the bounds 
        # (case data contains initials values in Vm - not used with IPOPT)
        Vm[i]=0.5*(b.Vmax+b.Vmin); 
        i=i+1
    end
    @assert i-1==length(opfdata.buses)

    # set all angles to the angle of the reference bus
    Va = opfdata.buses[opfdata.bus_ref].Va * ones(length(opfdata.buses))

    return Pg,Qg,Vm,Va
end

for casename in ["case9", "case118", "case300"]
    casepath = joinpath(dirname(@__FILE__), "data", casename)
    max_iter=100
    opfdata = opf_loaddata(casepath)
    Pg0, Qg0, Vm0, Va0 = initialPt_IPOPT(opfdata)
    opfmodel_ref, Pg_ref, Qg_ref, Va_ref, Vm_ref = model(opfdata; solver="Ipopt")
    opfmodel, Pg, Qg, Va, Vm = model(opfdata; solver="Hiop")
    opfmodel_ref ,status = solve(opfmodel_ref,opfdata)
    opfmodel,status = solve(opfmodel,opfdata)
    @test value.(Pg) ≈ value.(Pg)
    @test value.(Qg) ≈ value.(Qg)
    @test value.(Vm) ≈ value.(Vm)
    @test value.(Va) ≈ value.(Va)
    @test objective_value(opfmodel) ≈ objective_value(opfmodel_ref)
end