function compare_matrices(geometry,coil)
    # Compute matrices using CPU function
    K_cpu, MBr_cpu, MBz_cpu = CreationVectorPotentialMatrix12PancakeCoils(geometry)
    K_cpu, MBr_cpu, MBz_cpu =  ApplySymmetry12PancakeCoils_cpu(K_cpu, MBr_cpu, MBz_cpu, coil)
    
    # Compute matrices using CUDA function
    K_cuda, MBr_cuda, MBz_cuda = CreationVectorPotentialMatrix12PancakeCoils_CUDA
    (geometry)
    K_cuda, MBr_cuda, MBz_cuda = ApplySymmetry12PancakeCoils(K_cuda, MBr_cuda, MBz_cuda, coil)
    #K_cuda, MBr_cuda, MBz_cuda = CreationVectorPotentialMatrix12PancakeCoilsAdaptive(geometry)
    
    # Compute relative differences
    rel_diff_K = norm(K_cpu - K_cuda) / norm(K_cpu)
    rel_diff_MBr = norm(MBr_cpu - MBr_cuda) / norm(MBr_cpu)
    rel_diff_MBz = norm(MBz_cpu - MBz_cuda) / norm(MBz_cpu)
    
    println("Relative difference for K: ", rel_diff_K)
    println("Relative difference for MBr: ", rel_diff_MBr)
    println("Relative difference for MBz: ", rel_diff_MBz)
    
    # Plot the matrices
    plot_MBr_diff = heatmap(abs.((MBr_cpu - MBr_cuda) ./ MBr_cpu), title="Relative Difference in MBr", xlabel="Index", ylabel="Index")
    plot_MBz_diff = heatmap(abs.((MBz_cpu - MBz_cuda) ./ MBz_cpu), title="Relative Difference in MBz", xlabel="Index", ylabel="Index")
    plot_K_diff = heatmap(abs.((K_cpu - K_cuda) ./ K_cpu), title="Relative Difference in K", xlabel="Index", ylabel="Index")
    plot_K = heatmap(K_cuda, title="K CUDA values", xlabel="Index", ylabel="Index")
    plot_MBz = heatmap(MBz_cuda, title="MBz CUDA values", xlabel="Index", ylabel="Index")

    display(plot_MBr_diff)
    display(plot_MBz_diff)
    display(plot_K_diff)
    display(plot_K) 
    display(plot_MBz)
    
    return rel_diff_K, rel_diff_MBr, rel_diff_MBz
end