function PlotPowerCurve12PancakeCoils(coil)
    N_puntos_total = coil.Nr * coil.Nz * coil.Ne * coil.Nc
    T = coil.results.T
    Js = coil.results.Js
    E = coil.results.E
    V = coil.V[:]
    P = zeros(length(T))

    if !isequal(coil.Z_middle, coil.z_middle)  # Estamos ante una bobina de doble pancake
        for i = 1:length(T)
            P[i] = 2 * sum(Js[i, 1:N_puntos_total]' .* E(Js[i, 1:N_puntos_total])' .* V[1:N_puntos_total]')
        end
    else
        for i = 1:length(T)
            P[i] = sum(Js[i, 1:N_puntos_total]' .* E(Js[i, 1:N_puntos_total])' .* V[1:N_puntos_total])
        end
    end

    # Set x-ticks to show all integer values in the range
    xticks_vals = collect(floor(Int, minimum(T)):ceil(Int, maximum(T)))

    plt = PlotlyJS.Plot(
        [PlotlyJS.scatter(x=T, y=P, mode="lines", name="Power")],
        PlotlyJS.Layout(
            xaxis=attr(
                title="Time (s)",
                showgrid=true,
                gridcolor="lightgray",
                tickmode="array",
                tickvals=xticks_vals,
                ticktext=string.(xticks_vals)
            ),
            yaxis=attr(title="Power (W)", showgrid=true, gridcolor="lightgray"),
            showlegend=false,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
    )
    display(plt)

    # Save as PNG file
    try
        PlotlyJS.savefig(plt, "power_curve.png")
        println("Power curve plot saved as power_curve.png")
    catch err
        @warn "Could not save power curve plot as PNG: $err"
    end

    Q = trapz(T, P)
    println(Q)
    coil.results.Q = Q
    #coil.results.P = P
end