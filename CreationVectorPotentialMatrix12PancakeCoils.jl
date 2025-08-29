function integrand_K(θ_ac, ri, zi, aj, bj, cj, dj)
    ej = (aj .+ bj) ./ 2

	term1 = log.((sqrt.((zi .- cj).^2 .+ ri.^2 .+ aj.^2 .- 2 .*ri.*aj.*cos(θ_ac)) .+ zi .- cj) ./
		(sqrt.((zi .- dj).^2 .+ ri.^2 .+ aj.^2 .- 2 .*ri.*aj.*cos(θ_ac)) .+ zi .- dj)).* aj .* cos(θ_ac)

	term2 = 4 .* log.((sqrt.((zi .- cj).^2 .+ ri.^2 .+ ej.^2 .- 2 .*ri.*ej.*cos(θ_ac)) .+ zi .- cj) ./
		(sqrt.((zi .- dj).^2 .+ ri.^2 .+ ej.^2 .- 2 .*ri.*ej.*cos(θ_ac)) .+ zi .- dj)) .* ej .* cos(θ_ac)

	term3 = log.((sqrt.((zi .- cj).^2 .+ ri.^2 .+ bj.^2 .- 2 .*ri.*bj.*cos(θ_ac)) .+ zi .- cj) ./
		(sqrt.((zi .- dj).^2 .+ ri.^2 .+ bj.^2 .- 2 .*ri.*bj.*cos(θ_ac)) .+ zi .- dj)) .* bj .* cos(θ_ac)

    return (bj .- aj) ./ 6 .*(term1 .+ term2 .+ term3)
end

function integrand_K_diag(r_ac, θ_ac, ri, zi, aj, bj, cj, dj)	
	integrand = log((sqrt((zi-cj)^2+ri^2+r_ac^2-2*ri*r_ac*cos(θ_ac)) + zi - cj)/(sqrt((zi-dj)^2+ri^2+r_ac^2-2*ri*r_ac*cos(θ_ac)) + zi-dj))* r_ac*cos(θ_ac)
	return integrand
end

function integrand_MBz_dist(theta_ac, ri, zi, aj, bj, cj, dj)
	ej = (aj .+ bj) ./ 2
	integrand = (bj.-aj)/6 .*(
				(((aj.*(aj .- ri*sin(theta_ac)).*(dj .- zi))./((aj.^2 .- 2 .*sin(theta_ac).*aj.*ri .+ ri.^2).*((dj .- zi).^2 .+ aj.^2 .+ ri.^2 .- 2 .*aj.*ri.*sin(theta_ac)).^(1/2))
				.-(aj.*(aj .- ri*sin(theta_ac)).*(cj .- zi))./((aj.^2 .- 2 .*sin(theta_ac).*aj.*ri .+ ri.^2).*((cj .- zi).^2 .+ aj.^2 .+ ri.^2 .- 2 .*aj.*ri.*sin(theta_ac)).^(1/2))))
				.+4  .*(((ej.*(ej .- ri.*sin(theta_ac)).*(dj .- zi))./((ej.^2 .- 2 .*sin(theta_ac).*ej.*ri .+ ri.^2).*((dj .- zi).^2 .+ ej.^2 .+ ri.^2 .- 2 .*ej.*ri.*sin(theta_ac)).^(1/2))
				.-(ej.*(ej .- ri*sin(theta_ac)).*(cj .- zi))./((ej.^2 .- 2*sin(theta_ac).*ej.*ri .+ ri.^2).*((cj .- zi).^2 .+ ej.^2 .+ ri.^2 .- 2*ej.*ri.*sin(theta_ac)).^(1/2))))
				.+(((bj.*(bj .- ri*sin(theta_ac)).*(dj .- zi))./((bj.^2 .- 2 .*sin(theta_ac).*bj.*ri .+ ri.^2).*((dj .- zi).^2 .+ bj.^2 .+ ri.^2 .- 2 .*bj.*ri.*sin(theta_ac)).^(1/2))
				.-(bj.*(bj .- ri.*sin(theta_ac)).*(cj .- zi))./((bj.^2 .- 2 .*sin(theta_ac).*bj.*ri .+ ri.^2).*((cj .- zi).^2 .+ bj.^2 .+ ri.^2 .- 2 .*bj.*ri.*sin(theta_ac)).^(1/2)))))
	return integrand
end

function integrand_MBr_dist(θ_ac, ri, zi, aj, bj, cj, dj)
	ej = (aj .+ bj) ./ 2
	term1 = ((aj .* sin.(θ_ac)) ./ sqrt.((dj .- zi).^2 .+ aj.^2 .+ ri.^2 .- 2 .* aj .* ri .* sin.(θ_ac)) .- (aj .* sin.(θ_ac)) ./ sqrt.((cj .- zi).^2 .+ aj.^2 .+ ri.^2 .- 2 .* aj .* ri .* sin.(θ_ac)))

  term2 = 4 .* ((ej .* sin.(θ_ac)) ./ sqrt.((dj .- zi).^2 .+ ej.^2 .+ ri.^2 .- 			2 .* ej .* ri .* sin.(θ_ac)) .- (ej .* sin.(θ_ac)) ./ sqrt.((cj .- 				zi).^2 .+ ej.^2 .+ ri.^2 .- 2 .* ej .* ri .* sin.(θ_ac)))

  term3 = ((bj .* sin.(θ_ac)) ./ sqrt.((dj .- zi).^2 .+ bj.^2 .+ ri.^2 .- 2 .* 			bj .* ri .* sin.(θ_ac)) .- (bj .* sin.(θ_ac)) ./ sqrt.((cj .- zi).^2 .+ 		bj.^2 .+ ri.^2 .- 2 .* bj .* ri .* sin.(θ_ac)))
	return (bj .- aj) ./ 6 .* (term1 + term2 + term3)
end

function CreationVectorPotentialMatrix12PancakeCoils(geometry)
	N = length(geometry.r_middle[:])
    K = Matrix{Float64}(undef, N, N)
    MBr = similar(K)
    MBz = similar(K)

	# Preallocate temporary storage
    temp_K = Vector{Float64}(undef, N)
    temp_MBz = similar(temp_K)
    temp_MBr = similar(temp_K)

    aj = geometry.r_low[:]
    bj = geometry.r_high[:]
    cj = geometry.z_low[:]
    dj = geometry.z_high[:]

	# Loop through each element
	for i in 1:1:N
		ri = geometry.r_middle[:][i]
		zi = geometry.z_middle[:][i]

		temp_K  .= quadgk(θ_ac -> integrand_K(θ_ac, ri, zi, aj, bj, cj, dj), 0.0, pi, rtol = 1e-6, atol = 1e-9)[1]
		K[i, :] .= temp_K
		
		K[i, i] = hcubature(r -> integrand_K_diag(r[1], r[2], ri, zi, aj[i], bj[i], cj[i], dj[i]), [aj[i], 0.0], [bj[i], pi], rtol = 1e-6, atol= 1e-9)[1] 

		temp_MBz .= quadgk(θ_ac -> integrand_MBz_dist(θ_ac, ri, zi, aj, bj, cj, dj), -pi/2, pi/2, rtol = 1e-6, atol = 1e-9)[1]
		temp_MBr .= quadgk(θ_ac -> integrand_MBr_dist(θ_ac, ri, zi, aj, bj, cj, dj), -pi/2, pi/2, rtol = 1e-6, atol = 1e-9)[1]
		MBz[i, :] .= temp_MBz
		MBr[i, :] .= temp_MBr
		MBz[i, i] = 0
		MBr[i, i] = 0	
	end
	
	# In-place unit conversions and NaN handling
	@. K = real(K / 1e6) # K a m^2
	@. MBz = real(MBz / 1e3)
	@. MBr = real(MBr / 1e3)
	replace!(x -> isnan(x) | isinf(x) ? 0.0 : x, K)
	replace!(x -> isnan(x) | isinf(x) ? 0.0 : x, MBz)
	replace!(x -> isnan(x) | isinf(x) ? 0.0 : x, MBr)

	return K, MBr, MBz
end