function SetMeshSuperconductor2PancakeCoils(coil::Coil)
	h = coil.h * 1e3 # altura [mm]
	w = coil.w * 1e3 # radio [mm]
	Nr, Nz = coil.Nr, coil.Nz # número de elementos

	# Vector radial con elementos más pequeños cerca de los bordes.
	r_vector = (w*cos.(pi/2*(1 .- (0:Nr)/Nr)))'
	# Vector axial con elementos más pequeños cerca de la parte superior e inferior.
	z_vector_up = h/2 .- h/2*cos.(pi .- (0:Nz)/Nz*pi)' 
	z_vector_down = -reverse(z_vector_up)

	# Malla para una sola espira.
	r_grid = r_vector' .+ 0z_vector_up
	z_grid_up = 0*r_vector' .+ z_vector_up
	z_grid_down = 0*r_vector' .+ z_vector_down
	
	R_low, R_high, R_middle = zeros(Nz, coil.Ne), zeros(Nz, coil.Ne), zeros(Nz, coil.Ne)
    Z_high_up = hcat(map(x -> (z_grid_up[1:end-1, 1:end-1])', 1:coil.Ne)...)
    Z_low_up = hcat(map(x -> (z_grid_up[2:end, 2:end])', 1:coil.Ne)...)
    Z_middle_up = (Z_low_up .+ Z_high_up) / 2.0
	Z_high_down = hcat(map(x -> (z_grid_down[1:end-1, 1:end-1])', 1:coil.Ne)...)
    Z_low_down = hcat(map(x -> (z_grid_down[2:end, 2:end])', 1:coil.Ne)...)
    Z_middle_down = (Z_low_down .+ Z_high_down) ./ 2.0

	# Búcle para cada espira.
	for N_espira = 1:1:coil.Ne
		# Cálculo radio interno de cada espira.
		r_int = coil.r_int + coil.sep + (coil.w + 2*coil.sep)*(N_espira-1)
		r_int = 1000 .* r_int # [mm]

		# Límites y puntos medios de cada elemento.
		R_low[:,N_espira] = r_int .+ (r_grid[1:end-1, 1:end-1])'
	    R_high[:,N_espira] = r_int .+ (r_grid[2:end, 2:end])'
	    R_middle[:,N_espira] = (R_low[:,N_espira] + R_high[:,N_espira]) ./ 2.0
	end

	# Combinación datos de todas las espiras en un solo pancake.
  	r1_high = vcat(R_high, R_high)
	r1_low = vcat(R_low, R_low)
	r1_middle = vcat(R_middle, R_middle)
	
 	z1_high= vcat(Z_high_up, Z_high_down)
	z1_low=vcat(Z_low_up, Z_low_down)
	z1_middle=vcat(Z_middle_up, Z_middle_down)
	
	# Distancia entre los centros de los pancakes [mm].
 	z1 = 2*h + coil.d * 1e3 * (coil.Nc != 0)
	# Datos para todos los pancakes.
    z2_high = z1_high
    z2_low = z1_low
    z2_middle = z1_middle

	# Construcción del mesh de todas las bobinas.
	# Para cada bobina añade datos espiras.
	r2_high = vcat(map(x -> r1_high, 1:coil.Nc)...)
	r2_low = vcat(map(x -> r1_low, 1:coil.Nc)...)
	r2_middle = vcat(map(x -> r1_middle, 1:coil.Nc)...)
	
	# Iteración por cada bobina
	for i=2:1:coil.Nc
		z2_high = vcat((z1*(i-1)) .+ z1_high, z2_high)
	    z2_low = vcat((z1*(i-1)) .+ z1_low, z2_low)
	    z2_middle = vcat((z1*(i-1)) .+ z1_middle, z2_middle)
	end
	
	# Ajusta posición de cada pancake en base a distancia centro-a-centro.
	z_compensacion = (z2_middle[1]+z2_middle[length(z2_middle)])/2

	z2_high .-= z_compensacion
	z2_low .-= z_compensacion
	z2_middle .-= z_compensacion
	
	# Poner todas las espiras por encima del eje radial primero y después el resto.
	dimZ, dimR= size(z2_middle,1), size(z2_middle,2)

	z_middle = hcat(z2_middle[1:Int.(dimZ/2),:], z2_middle[Int.(dimZ/2)+1:dimZ,:])
	z_low = hcat(z2_low[1:Int.(dimZ/2),:], z2_low[Int.(dimZ/2)+1:dimZ,:])
	z_high = hcat(z2_high[1:Int.(dimZ/2),:], z2_high[Int.(dimZ/2)+1:dimZ,:])

	r_middle = hcat(r2_middle[1:Int.(dimZ/2),:], r2_middle[Int.(dimZ/2)+1:dimZ,:])
	r_low = hcat(r2_low[1:Int.(dimZ/2),:], r2_low[Int.(dimZ/2)+1:dimZ,:])
	r_high = hcat(r2_high[1:Int.(dimZ/2),:], r2_high[Int.(dimZ/2)+1:dimZ,:])

	coil.R_middle = r2_middle[1:Int.(dimZ/2),:]
	coil.Z_middle = z2_middle[1:Int.(dimZ/2),:]
	coil.r_high=r_high
	coil.r_low=r_low
	coil.r_middle=r_middle
	coil.z_high=z_high
	coil.z_low=z_low
	coil.z_middle=z_middle
	coil.s = ((r_high-r_low).*(z_high-z_low))/1e6 # cross-sectional area
	coil.V = pi*((r_high.^2-r_low.^2).*(z_high-z_low))/1e9 # volumen

	return coil
end