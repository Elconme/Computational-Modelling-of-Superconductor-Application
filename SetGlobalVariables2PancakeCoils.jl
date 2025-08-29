#import Pkg
#Pkg.add("QuadGK")
#Pkg.add("HCubature")
#Pkg.add("Debugger")
#Pkg.add("JuliaInterpreter")
#Pkg.add("CUDA")
#Pkg.add("MethodAnalysis")
#Pkg.add("JLD2")
#Pkg.add("Sundials")
#Pkg.add("LoopVectorization")
#Pkg.add("ThreadsX")
#Pkg.add("Plots")
#Pkg.add("Trapz")
#Pkg.add("PyPlot")
#Pkg.add("PythonCall")
#Pkg.add("MAT")
#Pkg.add("PlotlyJS")
###################################
#Pkg.add("DifferentialEquations")
#Pkg.add("MATLABDiffEq")
#################################
# Pkg.add("ModelingToolkit")
# Pkg.add("DifferentialEquations")
#Pkg.add("LinearAlgebra")

# 
# Pkg.add("SparseArrays")
# 
# Pkg.add("StructArrays")
# Pkg.add("RuntimeGeneratedFunctions")

# Pkg.add("Printf")
# Pkg.add("Statistics")
# Pkg.add("OrdinaryDiffEq")
#Pkg.add("DiffEqGPU")
# Pkg.add("Profile")
# Pkg.add("ProfileView")
# Pkg.add("DelimitedFiles")
#

using QuadGK
using HCubature
using Debugger
using JuliaInterpreter, MethodAnalysis
using JLD2
using CUDA
using CUDA.CUBLAS
using CUDA.CUSPARSE
using Sundials
using LoopVectorization
using ThreadsX
using Plots
using Trapz
using DifferentialEquations
using LinearAlgebra
using PlotlyJS
using DiffEqGPU


union!(JuliaInterpreter.compiled_modules, child_modules(Base))  # Exclude all Base submodules[6][31]
using SparseArrays
  # For saving/loading Julia objects

# using StructArrays
# using RuntimeGeneratedFunctions  # Add this at top



using Statistics  # For cumtrapz
using Printf 
#using OrdinaryDiffEq
#using DiffEqGPU

using DelimitedFiles

#PyPlot.matplotlib[:use]("TkAgg")=#

#plotlyjs()

mutable struct results
	T::Vector{Float64}  
    Js::Matrix{Float64} 
    Jc::Function
	E::Function
	Br::Matrix{Float64}
	Bz::Matrix{Float64}	
    Q::Float64
end

mutable struct Coil
    h::Float64 # Altura de la cinta superconductora
    Sep::Float64 # Grosor de la cinta entera
    w::Float64 # Espesor de la capa supercondcutora
    r_int::Float64 # Radio interno de la(s) bobina(s)
    Ne::Int64 # Número de espiras/bobina
    Nc::Int64 # Número de bobinas
    Nz::Int64 # Número de particiones en las que se divide cada cinta a lo largo de su altura
	Nr::Int64 # Número de particiones en las que se divide cada cinta a lo largo de su dirección radial
    d::Float64 # Distancia vertical entre bobinas. Hueco de aire entre dos bobinas adyacentes.
    sep::Float64 # Distancia entre el radio interno de una cinta y donde empieza la zona superconductora
    Ec::Float64 # Campo eléctrico crítico del superconductor
    n::Int64 # Exponente del superconductor
    jc # Parámetros de la densidad de corriente crítica
	# Propiedades del transitorio
    gamma::Float64 # Cuanto más grande, más precisión para fijar la corriente establecida. Demasiado grande-> Problemas de convergencia. Valores óptimos: 100-10000
    Iset::Function # Intensidad del transitorio
    t # Tiempo del transitorio
	r_ext::Float64 # Radio externo bobina
	mu0::Float64 # Constante

	Ks::Matrix{Float64} 
	MBrs::Matrix{Float64} 
	MBzs::Matrix{Float64} 
	Ss_super::Matrix{Float64} 

	R_middle::Matrix{Float64} 
	Z_middle::Matrix{Float64}
	
	r_high::Matrix{Float64}
	r_low::Matrix{Float64}
	r_middle::Matrix{Float64}
	
	z_high::Matrix{Float64}
	z_low::Matrix{Float64}
	z_middle::Matrix{Float64}
	
	s::Matrix{Float64} # cross-sectional area
	V::Matrix{Float64} # volumen  
	
	results::results

    function Coil(h, Sep, w, r_int, Ne, Nc, Nz, Nr, d, sep, Ec, n, jc, gamma, Iset, t, r_ext, mu0)
        # Calculate total elements
        N_total = Nz * Ne * Nc * Nr  # 2 accounts for upper/lower pancake sections
        
        new(
            h, Sep, w, r_int, Ne, Nc, Nz, Nr, d, sep, Ec, n, jc, 
            gamma, Iset, t, r_ext, mu0,
            # Initialize matrices with dynamic sizes
            zeros(N_total, N_total),  # Ks
            zeros(N_total, N_total),  # MBrs
            zeros(N_total, N_total),  # MBzs
            zeros(N_total, N_total),  # Ss_super
            zeros(Nz*Nc, 2*Ne),      # R_middle (2*Nz*Nc rows from upper/lower sections)
            zeros(Nz*Nc, 2*Ne),        # Z_middle
            zeros(Nz*Nc, 2*Ne), # r_high
		    zeros(Nz*Nc, 2*Ne), # r_low
		    zeros(Nz*Nc, 2*Ne), # r_middle
		    zeros(Nz*Nc, 2*Ne), # z_high
		    zeros(Nz*Nc, 2*Ne), # z_low
		    zeros(Nz*Nc, 2*Ne), # z_middle
		    zeros(Nz*Nc, 2*Ne), # s
		    zeros(Nz*Nc, 2*Ne), # V 
			results(Vector{Float64}(undef, 0), Matrix{Float64}(undef, 0, 0), x -> 0.0, x -> 0.0, Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0), 0.0) # Initialize results           
        )
    end
	
end

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