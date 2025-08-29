function PlotDistribution2PancakeCoils(coil, rep, animate, instante)
    # rep: 1: J/Jc
    #      2 Br
    #      3 Bz
    #      4 Heat

    # animate: 1: animate
    #          0: frame
    # instante :valor de instante a representar

    # Definicion de variables
    R_middle = coil.R_middle
    Z_middle = coil.Z_middle
    Js = coil.results.Js
    T = coil.results.T
    V = coil.V
    h = coil.h
    w = coil.w
    mu0 = 4 * pi * 1e-7
    MBzs = coil.MBzs
    MBrs = coil.MBrs
    c1 = coil.jc.c1
    c1_5_3 = (1 / c1)^(5 / 3)
    k0 = coil.jc.k0
    b0 = coil.jc.b0
    a0 = coil.jc.a0
    k1 = coil.jc.k1
    b1 = coil.jc.b1
    a1 = coil.jc.a1
    fi1 = coil.jc.fi1
    Ec = coil.Ec
    n = coil.n
    Ne = coil.Ne
    size_R1 = size(R_middle, 1)
    size_R2 = size(R_middle, 2)

    function Jc_val(js)
        B_val = sqrt.((mu0 / (2 * pi) .* (MBzs * js)).^2 + (mu0 / (2 * pi) .* (MBrs * js)).^2)
        w1 = c1 .* (B_val .+ c1_5_3).^(3 / 5)
        ang = atan.(MBrs * js, MBzs * js)
        Ic_val = k0 ./ ((B_val .+ b0).^a0) .+ k1 ./ ((B_val .+ b1).^a1) .* (w1.^2 .* cos.(ang .- fi1).^2 .+ sin.(ang .- fi1).^2).^(-1 / 2)
        return Ic_val ./ (h * w)
    end

    function E_val(js)
        jc = Jc_val(js)
        return Ec .* sign.(js) .* abs.(js ./ jc).^n
    end

    if mod(coil.Nc, 2) == 1 # Impar
        pos = size(Z_middle, 1) .- (coil.Nz:2 * coil.Nz:coil.Nz * (coil.Nc - 2))
        if coil.Nc == 1
            pos = []
        end
    else
        pos = 2 .* (coil.Nz:coil.Nz:coil.Nz * (coil.Nc / 2 - 1))
    end

    # Resultados de la simulacion
    Br_repre = zeros(length(coil.t), size(R_middle, 1) + 2 * length(pos), size(R_middle, 2))
    Bz_repre = zeros(length(coil.t), size(Z_middle, 1) + 2 * length(pos), size(Z_middle, 2))
    P_distribution_repre = zeros(length(coil.t), size(Z_middle, 1) + 2 * length(pos), size(Z_middle, 2))
    JsJc_repre = zeros(length(coil.t), size(Z_middle, 1) + 2 * length(pos), size(R_middle, 2))
    Br = reshape(coil.results.Br, (length(coil.t), size(R_middle, 1), size(R_middle, 2)))
    Bz = reshape(coil.results.Bz, (length(coil.t), size(R_middle, 1), size(R_middle, 2)))
    JsJc = reshape(Js ./ Jc_val(Js')', (length(coil.t), size(R_middle, 1), size(R_middle, 2)))
    Js_force = reshape(Js, (length(coil.t), size(R_middle, 1), size(R_middle, 2)))

    P_distribution = zeros(length(T), size(R_middle, 1), size(R_middle, 2))
    N_puntos_total = coil.Nc * coil.Ne * coil.Nz

    for i = 1:length(T)
        P_distribution[i, :, :] = reshape(Js[i, 1:N_puntos_total]' .* E_val(Js[i, 1:N_puntos_total])' .* V[1:N_puntos_total]', (size_R1, size_R2))
    end

    R_middle = coil.R_middle
    Z_middle = coil.Z_middle

    # Bobinas de doble pancake
    R = zeros(size(R_middle, 1) + 2 * length(pos), size(R_middle, 2))
    Z = zeros(size(Z_middle, 1) + 2 * length(pos), size(Z_middle, 2))
    j = 0

    for i = 1:size(Z_middle, 1)
        if i in pos
            Z[i+j, :] = Z_middle[i, :]
            R[i+j, :] = R_middle[i, :]
            Br_repre[:, i+j, :] = Br[:, i, :]
            Bz_repre[:, i+j, :] = Bz[:, i, :]
            JsJc_repre[:, i+j, :] = JsJc[:, i, :]
            P_distribution_repre[:, i+j, :] = P_distribution[:, i, :]
            j += 2
        else
            Z[i+j, :] = Z_middle[i, :]
            R[i+j, :] = R_middle[i, :]
            Br_repre[:, i+j, :] = Br[:, i, :]
            Bz_repre[:, i+j, :] = Bz[:, i, :]
            JsJc_repre[:, i+j, :] = JsJc[:, i, :]
            P_distribution_repre[:, i+j, :] = P_distribution[:, i, :]
        end
    end

    vec = findall(Z[:, 1] .== 0)
    for k = 1:2:length(vec)
        i = vec[k]
        Z[i, :] = Z[i-1, :] .- (Z[i-1, :] .- Z[i+2, :]) ./ 50
        Z[i+1, :] = Z[i+2, :] .+ (Z[i-1, :] .- Z[i+2, :]) ./ 50
        R[i, :] = R[i-1, :]
        R[i+1, :] = R[i+2, :]
    end

    function cumtrapz(x::AbstractVector, y::AbstractArray, dim::Integer)
        ndims_y = ndims(y)
        if !(1 <= dim <= ndims_y)
            throw(DimensionMismatch("Dimension $dim is invalid for an array with $(ndims_y) dimensions."))
        end
        size_y_dim = size(y, dim)
        if length(x) != size_y_dim
            throw(DimensionMismatch("Length of x ($length(x)) must match the size of y along dimension dim ($size_y_dim)."))
        end
        out_size = size(y)
        out = zeros(eltype(y), out_size)
        for i in 2:size_y_dim
            dx = x[i] - x[i-1]
            prev_y_slice = slice_along_dim(y, dim, i - 1)
            curr_y_slice = slice_along_dim(y, dim, i)
            out = add_to_along_dim(out, dim, i, dx .* (prev_y_slice .+ curr_y_slice) ./ 2)
        end
        return out
    end

    function slice_along_dim(arr::AbstractArray, dim::Integer, index::Integer)
        inds = ntuple(d -> d == dim ? index : Colon(), ndims(arr))
        return view(arr, inds...)
    end

    function add_to_along_dim(arr::AbstractArray, dim::Integer, index::Integer, values)
        inds = ntuple(d -> d == dim ? index : Colon(), ndims(arr))
        arr[inds...] .= arr[inds...] .+ values
        return arr
    end

    if rep == 1
        Y = JsJc_repre
    elseif rep == 2
        Y = Br_repre
    elseif rep == 3
        Y = Bz_repre
    elseif rep == 4
        Y = cumtrapz(T, P_distribution_repre, 1)
    end

    if size(R_middle, 2) == 1
        R = [R .- coil.w / 2, R .+ coil.w / 2]
        Z = [Z, Z]
        Y1 = zeros(length(coil.t), size(R, 1), 2)
        for i = 1:2
            Y1[:, :, i] = Y[:, :, :]
        end
        Y = Y1
    end

    if animate == 1
        Iset = coil.Iset
        xlabel!("r [mm]")
        ylabel!("z [mm]")
        lim = abs(maximum([abs(minimum(Y[:])) - 10e-12, maximum(Y[:]) + 10e-12]))
        clim = (-lim, lim)
        for i = 6:5:length(T)
            figure()
            colorbar()
            xlabel("r [mm]")
            ylabel("z [mm]")
            show()
            sleep(0.05)
        end
    else
        instante_index = round(Int, length(coil.t) / (T[end] - 0) * instante)
        i = max(1, min(instante_index, length(coil.t)))
        Iset = coil.Iset
        lim = abs(maximum([abs(minimum(Y[:])) - 10e-12, maximum(Y[:]) + 10e-12]))
        if size(R_middle, 2) == 1
            clim = (-lim * 0.99, lim * 1.01)
            xlim = (minimum(R_middle[1, :]) - coil.w / 2, maximum(R_middle[1, :]) + coil.w / 2)
        else
            clim = (-lim, lim)
            xlim = (minimum(R_middle[1, :]), maximum(R_middle[1, :]))
        end

        trace = PlotlyJS.heatmap(
            x=R[1, :],
            y=Z[:, 1], 
            z=reshape(Y[i, :, :], (size(R, 1), size(R, 2))),
            colorscale=[[0, "blue"], [0.5, "white"], [1, "red"]],
            zmin=clim[1],
            zmax=clim[2],
            zsmooth="best",
            connectgaps=true,
            showscale=true
        )
        layout = PlotlyJS.Layout(
            title=PlotlyJS.attr(text="Heat", x=0.5),
            xaxis=PlotlyJS.attr(title="r [mm]", range=xlim, showgrid=false, zeroline=false),
            yaxis=PlotlyJS.attr(title="z [mm]", range=(0, maximum(Z_middle[:, 1])), showgrid=false, zeroline=false),
            colorbar=PlotlyJS.attr(title="Intensity"),
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        p = PlotlyJS.Plot(trace, layout)
        display(p)

        # Save as PNG file
        try
            PlotlyJS.savefig(p, "Jc.png")
            println("Jc plot saved")
        catch err
            @warn "Could not save Jc plot as PNG: $err"
        end
    end
end