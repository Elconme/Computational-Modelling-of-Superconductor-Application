function Iset_function(t)
    return 600*sin(2*pi*t)#700 * t
end

function SetGlobalVariables2PancakeCoils()
    coil = Coil(        
        4.8e-3,  # h
        340e-6, # Sep
        33e-6,  # w
        172e-3, # r_int
        144,  # Ne 
        2,  # Nc
        10,  # Nz
		1, # Nr
        20e-3, # d
        (340e-6 - 33e-6)/2, # sep
        1e-4,  # Ec
        40,  # n
        (k0 = 3.071514313357327e+04, k1 = 18500, a0 = 1.3, a1 = 0.809, b0 = 13.8, b1 = 13.8, fi1 = -0.18, c1 = 2.15),  # jc
        100,  # gamma
        Iset_function,
        0:0.1:10,  # t
		0, # r_ext
		0, # mu0,
    )

    # Llamada a la función para calcular variables dependientes
    coil = SetGlobalDependentVariables(coil, 1)

    return coil
end

function SetGlobalDependentVariables(coil::Coil, gpu)
    coil.r_ext = 270e-3;#coil.r_int + coil.Sep * coil.Ne # Radio externo 
    coil.mu0 = 4π * 1e-7  
    coil.Nr = 1 # Número de particiones en las que se divide cada cinta a lo largo de su dirección radial
    @time coil = SetMeshSuperconductor2PancakeCoils(coil)
    @time geometry = BuildGeometry12PancakeCoils(coil)
    if gpu
        @time K, MBr, MBz = CreationVectorPotentialMatrix12PancakeCoils_CUDA(geometry)
        @time K, MBr, MBz = ApplySymmetry12PancakeCoils(K, MBr, MBz, coil)
    else
        @time K, MBr, MBz = CreationVectorPotentialMatrix12PancakeCoils(geometry)
        @time K, _, _ = ApplySymmetry12PancakeCoils_cpu(K, MBr, MBz, coil)
    end
    @time S_super = CreationDphiMatrix2PancakeCoils(geometry, coil)

    return coil
end