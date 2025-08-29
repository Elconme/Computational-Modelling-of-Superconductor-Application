function SolveODEProblem2PanckakeCoilsCUDA(coil)
    # ==== Parámetros ====
    Ks_d = CuArray(coil.Ks)
    MBrs_d = CuArray(coil.MBrs)
    MBzs_d = CuArray(coil.MBzs)
    Ss_super_d = CuArray(coil.Ss_super)
    n, Ec, mu0, gamma, h, w = coil.n, coil.Ec, coil.mu0, coil.gamma, coil.h, coil.w
    Iset = coil.Iset
    N_total = coil.Nz * coil.Ne * coil.Nc * coil.Nr
    tspan = (0.0, 1.0)
    
    # ==== Parámetros Jc ====
    k0, k1, a0, a1, b0, b1, fi1, c1 = coil.jc.k0, coil.jc.k1, coil.jc.a0, coil.jc.a1, coil.jc.b0, coil.jc.b1, coil.jc.fi1, coil.jc.c1
    c1_5_3 = (1 / c1)^(5 / 3)
    
    # ==== Inicialización ====
    J0_d = CuArray(zeros(Float64, N_total))
    ones_buf_d = CuArray(ones(Float64, N_total))
    
    # ==== Precalcular Iset en GPU ====
    t_points = collect(tspan[1]:0.1:tspan[2])
    Iset_gpu = CuArray(Iset.(t_points))

    function safe_division(num, den, epsilon=1e-12)
        return num ./ max.(den, epsilon)
    end

    function fun_cuda!(dJs_d, Js_d, p, t)
        MBzs_Js = MBzs_d * Js_d
        MBrs_Js = MBrs_d * Js_d
    
        B_val = sqrt.((mu0 / (2π) .* MBzs_Js).^2 .+ (mu0 / (2π) .* MBrs_Js).^2)
        w1 = c1 .* (B_val .+ c1_5_3).^(3 / 5)
        ang = atan.(MBrs_Js, MBzs_Js)
    
        Ic_val = k0 ./ ((B_val .+ b0).^a0) .+ k1 ./ ((B_val .+ b1).^a1) .* 
                 safe_division(w1.^2 .* cos.(ang .- fi1).^2 .+ sin.(ang .- fi1).^2, 1.0)
        Jc_val = safe_division(Ic_val, h * w)
    
        E_val = Ec .* sign.(Js_d) .* abs.(safe_division(Js_d, Jc_val)).^n
    
        # Obtenemos Iset(t) directamente en CPU
        Iset_val = Iset(t)  # asume que Iset es una función
    
        # Broadcast de escalar CPU sobre vector GPU: seguro y eficiente
        dphi_val = gamma .* (Ss_super_d * Js_d .- Iset_val .* ones_buf_d)
    
        @. dJs_d = -5e6 * (E_val + dphi_val)
        return nothing
    end
    

    function jac_cuda!(Jmat_d, Js_d, p, t)
        MBzs_Js = MBzs_d * Js_d
        MBrs_Js = MBrs_d * Js_d

        B_val = sqrt.((mu0 / (2π) .* MBzs_Js).^2 .+ (mu0 / (2π) .* MBrs_Js).^2)
        w1 = c1 .* (B_val .+ c1_5_3).^(3 / 5)
        ang = atan.(MBrs_Js, MBzs_Js)

        Ic_val = k0 ./ ((B_val .+ b0).^a0) .+ k1 ./ ((B_val .+ b1).^a1) .* 
                 safe_division(w1.^2 .* cos.(ang .- fi1).^2 .+ sin.(ang .- fi1).^2, 1.0)
        Jc_val = safe_division(Ic_val, h * w)

        dEdJ_val = Ec .* (n ./ Jc_val) .* abs.(safe_division(Js_d, Jc_val)).^(n - 1)

        # Inicializar y rellenar diagonal
        fill!(Jmat_d, 0.0)
        Jmat_d .+= -5e6 .* gamma .* Ss_super_d
        I_d = CuArray(I, N_total, N_total) # matrix identidad en GPU
        Jmat_d .+= -5e6 .* dEdJ_val .* I_d  # broadcasting implícito con la identidad

        return nothing
    end

    f = ODEFunction(fun_cuda!; jac=jac_cuda!, mass_matrix=Ks_d)
    prob = ODEProblem(f, J0_d, tspan)
    monteprob_jac = EnsembleProblem(prob)

    #sol = solve(prob, Rodas5P(), EnsembleGPUKernel(CUDA.CUDABackend()), abstol=1e-9, reltol=1e-5, saveat=0:0.1:1.0)
    sol = solve(monteprob_jac, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend()), trajectories = 10_000, saveat = 0.1f0)
    #sol = solve(prob, Rodas5P(), abstol=1e-9, reltol=1e-5, saveat=0:0.1:1.0)
    
    # ==== Extraer resultados ====
    T = sol.t
    Js_d = sol.u
    Js = Matrix(reduce(hcat, Js_d)')  # (steps, N_total)

    function Jc_func(Js)
        B_val = hypot.(mu0 / (2π) .* (coil.MBzs * Js), mu0 / (2π) .* (coil.MBrs * Js))
        w1 = c1 .* (B_val .+ c1_5_3).^(3 / 5)
        ang = atan.(coil.MBrs * Js, coil.MBzs * Js)
        Ic_val = k0 ./ ((B_val .+ b0).^a0) .+ k1 ./ ((B_val .+ b1).^a1) .* 
                 safe_division(w1.^2 .* cos.(ang .- fi1).^2 .+ sin.(ang .- fi1).^2, 1.0)
        safe_division(Ic_val, h * w)
    end

    function E_func(Js)
        Jc_val = Jc_func(Js)
        Ec .* sign.(Js) .* abs.(safe_division(Js, Jc_val)).^n
    end

    Br = (mu0 / (2π) .* (coil.MBrs * Js'))'
    Bz = (mu0 / (2π) .* (coil.MBzs * Js'))'

    coil.results.T = T
    coil.results.Js = Js
    coil.results.Jc = Jc_func
    coil.results.E = E_func
    coil.results.Br = Br
    coil.results.Bz = Bz
end