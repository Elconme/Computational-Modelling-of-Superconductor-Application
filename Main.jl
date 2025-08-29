# Cálculo de las pérdidas en 2PancakeCoils
@time coil = SetGlobalVariables2PancakeCoils()
@time SolveODEProblem2PanckakeCoils(coil)
#@save "coil_data.jld2" coil
#@load "coil_data.jld2" coil
PlotPowerCurve12PancakeCoils(coil)
PlotDistribution2PancakeCoils(coil,4,0,1)

#=begin
    @time coil = SetGlobalVariables2PancakeCoils()
    coil_cpu = deepcopy(coil)
    coil_gpu = deepcopy(coil)

    println("Running CPU solver...")
    @time SolveODEProblem2PanckakeCoils(coil_cpu)

    println("Running GPU solver...")
    @time SolveODEProblem2PanckakeCoilsCUDA(coil_gpu)

    # Compare results
    compare_results(coil_cpu, coil_gpu)
end=#

#=begin
    @time coil, geometry = SetGlobalVariables2PancakeCoils() 
    @time compare_matrices(geometry, coil)
end=#