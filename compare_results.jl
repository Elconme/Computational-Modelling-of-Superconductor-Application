function compare_results(coil_cpu, coil_gpu)
    println("Comparing results between CPU and GPU solvers...")

    # Compare Js
    println("Comparing Js...")
    diff_Js = norm(coil_cpu.results.Js - coil_gpu.results.Js) / norm(coil_cpu.results.Js)
    println("Relative difference in Js: ", diff_Js)

    # Compare Br
    println("Comparing Br...")
    diff_Br = norm(coil_cpu.results.Br - coil_gpu.results.Br) / norm(coil_cpu.results.Br)
    println("Relative difference in Br: ", diff_Br)

    # Compare Bz
    println("Comparing Bz...")
    diff_Bz = norm(coil_cpu.results.Bz - coil_gpu.results.Bz) / norm(coil_cpu.results.Bz)
    println("Relative difference in Bz: ", diff_Bz)

    return diff_Js, diff_Br, diff_Bz, diff_Jc, diff_E
end