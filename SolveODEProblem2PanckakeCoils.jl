function SolveODEProblem2PanckakeCoils(coil)
    # Extract parameters from coil structure
    Ks = coil.Ks
    MBrs = coil.MBrs
    MBzs = coil.MBzs
    Ss_super = coil.Ss_super
    n = coil.n
    Ec = coil.Ec
    mu0 = coil.mu0
    gamma = coil.gamma
    Iset = coil.Iset # This is a function of time, Iset(t)
    h = coil.h
    w = coil.w
    Nz = coil.Nz
    Ne = coil.Ne
    Nc = coil.Nc
    Nr = coil.Nr
    N_total = Nz * Ne * Nc * Nr
    tspan = (coil.t[1], coil.t[end])

    # Extract Jc parameters from coil.jc
    k0, k1, a0, a1, b0, b1, fi1, c1 = coil.jc.k0, coil.jc.k1, coil.jc.a0, coil.jc.a1, coil.jc.b0, coil.jc.b1, coil.jc.fi1, coil.jc.c1
    c1_5_3 = (1 / c1)^(5 / 3)
    mu0_2pi = mu0 / (2π)

    # Initial current distribution
    J0 = zeros(N_total)

    # Preallocate temporary arrays for in-place operations
    MBzs_Js = zeros(N_total)
    MBrs_Js = zeros(N_total)
    ones_buf = ones(N_total)

    # Define a small epsilon for numerical stability
    epsilon = 1e-12

    # Inline functions using closures for efficiency
    compute_B_val = (MBzs_Js_vec, MBrs_Js_vec) -> @. hypot(mu0_2pi * MBzs_Js_vec, mu0_2pi * MBrs_Js_vec)

    compute_Ic_val = (B_val, ang) -> begin
        w1 = @. c1 * (B_val + c1_5_3)^(3 / 5)
        @. k0 / max((B_val + b0)^a0, epsilon) + k1 / max((B_val + b1)^a1, epsilon) *
            (w1^2 * cos(ang - fi1)^2 + sin(ang - fi1)^2)^(-0.5)
    end
    compute_Jc_val = Ic_val -> @. Ic_val / max(h * w, epsilon)
    compute_E_val = (Js_vec, Jc_val) -> @. Ec * sign(Js_vec) * abs(Js_vec / max(Jc_val, epsilon))^n
    compute_dphi_val = (Ss_super_mat, Js_vec, Iset_val, gamma, ones_buf_vec) ->
        gamma * (Ss_super_mat * Js_vec - Iset_val * ones_buf_vec)

    function fun!(dJs, Js, p, t)
        # Calculate intermediate values
        mul!(MBzs_Js, MBzs, Js)
        mul!(MBrs_Js, MBrs, Js)

        B_val = compute_B_val(MBzs_Js, MBrs_Js)
        ang = atan.(MBrs_Js, MBzs_Js)
        Ic_val = compute_Ic_val(B_val, ang)
        Jc_val = compute_Jc_val(Ic_val)
        E_val = compute_E_val(Js, Jc_val)
        Iset_t_val = Iset(t)
        dphi_val = compute_dphi_val(Ss_super, Js, Iset_t_val, gamma, ones_buf)

        # Update derivative in-place
        @. dJs = -5e6 * (E_val + dphi_val)
        return nothing
    end

    function jac!(Jmat, Js, p, t)
        # Calculate intermediate values
        mul!(MBzs_Js, MBzs, Js)
        mul!(MBrs_Js, MBrs, Js)
        B_val = compute_B_val(MBzs_Js, MBrs_Js)
        ang = atan.(MBrs_Js, MBzs_Js)
        Ic_val = compute_Ic_val(B_val, ang)
        Jc_val = compute_Jc_val(Ic_val)

        dEdJ_direct_val = @. Ec * (n / max(Jc_val, epsilon)) * abs(Js / max(Jc_val, epsilon))^(n - 1)

        fill!(Jmat, 0.0)
        for i in eachindex(dEdJ_direct_val)
            Jmat[i, i] = -5e6 * dEdJ_direct_val[i]
        end

        mul!(Jmat, -5e6 * gamma, Ss_super, 1.0, 1.0)

        return nothing
    end

    # Solve the ODE problem
    mass_matrix_val = real.(Ks)

    f = ODEFunction(fun!; jac=jac!, mass_matrix=mass_matrix_val)
    prob = ODEProblem(f, J0, tspan)

    # Consider adjusting tolerances if initial warning persists
    sol = solve(prob, ROS34PW2(), abstol=1e-9, reltol=1e-5, saveat=coil.t)

    # Extract results
    T = sol.t
    Js = sol.u
    Js_out = reshape(hcat(Js...), N_total, :)'

    Jc_func = Js_vec -> begin
        B_val = hypot.(mu0_2pi .* (MBzs * Js_vec), mu0_2pi .* (MBrs * Js_vec))
        w1 = c1 .* (B_val .+ c1_5_3).^(3/5)
        ang = atan.(MBrs * Js_vec, MBzs * Js_vec)
        # Correctly using epsilon here
        Ic_val = k0 ./ ((B_val .+ b0).^a0 .+ epsilon) .+ k1 ./ ((B_val .+ b1).^a1 .+ epsilon) .*
                (w1.^2 .* cos.(ang .- fi1).^2 .+ sin.(ang .- fi1).^2).^(-1/2)
        Ic_val ./ (h * w + epsilon)
    end

    E_func = Js_vec -> Ec .* sign.(Js_vec) .* abs.(Js_vec ./ (Jc_func(Js_vec) .+ epsilon)).^n

    Br = mu0_2pi .* (MBrs * Js_out')'
    Bz = mu0_2pi .* (MBzs * Js_out')'

    # Save results
    coil.results.T = T
    coil.results.Js = Js_out
    coil.results.Jc = Jc_func
    coil.results.E = E_func
    coil.results.Br = Br
    coil.results.Bz = Bz
    return coil
end