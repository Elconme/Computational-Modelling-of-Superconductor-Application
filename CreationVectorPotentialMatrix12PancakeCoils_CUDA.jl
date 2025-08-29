function integrand_K_diag(r_ac, θ_ac, ri, zi, aj, bj, cj, dj)	
	integrand = log((sqrt((zi-cj)^2+ri^2+r_ac^2-2*ri*r_ac*cos(θ_ac)) + zi - cj)/(sqrt((zi-dj)^2+ri^2+r_ac^2-2*ri*r_ac*cos(θ_ac)) + zi-dj))* r_ac*cos(θ_ac)
	return integrand
end

function CreationVectorPotentialMatrix12PancakeCoils_CUDA(geometry)
    Nz = geometry.Nz
    Nr = geometry.Nr
    Nc = geometry.Nc
    Ne = geometry.Ne
    N = Ne * Nz * Nr * Nc
        if !CUDA.functional()
        error("CUDA GPU is not available")
    end

    # Transfer geometry data to GPU (optimized for ri)
    ri_unique_row = CuArray(reshape(geometry.r_middle, :,1)) # Transfer the first Ne values of the first row
    zi = CuArray(reshape(geometry.z_middle, :, 1))
    aj = CuArray(reshape(geometry.r_low, 1, :))
    bj = CuArray(reshape(geometry.r_high, 1, :))
    cj = CuArray(reshape(geometry.z_low, 1, :))
    dj = CuArray(reshape(geometry.z_high, 1, :))

    # Initialize output matrices with the desired dimensions 400x800
    K = CUDA.zeros(Float64, Ne * Nz * Nc * Nr, 2 * Ne * Nc * Nz * Nr)
    MBr = CUDA.zeros(Float64, Ne * Nz * Nr * Nc, 2 * Ne * Nc * Nz * Nr)
    MBz = CUDA.zeros(Float64, Ne * Nz * Nr * Nc, 2 * Ne * Nc * Nz * Nr)

    # Define integration points
    n_points_low = 4000
    n_points_high_K = 4000
    n_points_high_M = 4000
    θ_range_K_low = collect(range(0, π, length=n_points_low))
    θ_range_M_low = collect(range(-π/2, π/2, length=n_points_low))
    θ_range_K_high = collect(range(0, π, length=n_points_high_K))
    θ_range_M_high = collect(range(-π/2, π/2, length=n_points_high_M))
    dθ_K_low = π / n_points_low
    dθ_M_low = π / n_points_low
    dθ_K_high = π / n_points_high_K
    dθ_M_high = π / n_points_high_M

    n_points_r = 1000
    # Compute r_range for each pair of aj and bj
    r_ranges_gpu = create_r_ranges_gpu(aj, bj, n_points_r)
    dr = CuArray((bj .- aj) ./ n_points_r)

    # Transfer integration points to GPU
    θ_range_K_low = CuArray(θ_range_K_low)
    θ_range_M_low = CuArray(θ_range_M_low)
    θ_range_K_high = CuArray(θ_range_K_high)
    θ_range_M_high = CuArray(θ_range_M_high)

    # Define regions for high n_points
    region1_start = 1
    region1_end = Ne * Nz * Nr * Nc/2  # Adjusted region end
    region2_start = Ne * Nz * Nr * Nc/2 + 1 # Adjusted region start
    region2_end = Ne * Nz * Nr * Nc # Adjusted region end

    # Define safe logarithm inline logic
    # @inline function safe_log(num::Float64, den::Float64)
    #     epsilon = 1.0e-10
    #     safe_num = max(num, epsilon)
    #     safe_den = max(den, epsilon)
    #     return log(max(min(safe_num / safe_den, 1.0e30), 1.0e-30))
    # end

    # CUDA kernel for integration
    function cuda_kernel!(K, MBr, MBz, θ_range_K_low, θ_range_M_low, θ_range_K_high, θ_range_M_high, ri_unique_row, zi, aj, bj, cj, dj, N, dθ_K_low, dθ_M_low, dθ_K_high, dθ_M_high, region1_start, region1_end, region2_start, region2_end, dr, r_ranges_gpu)
        idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

        if idx <= Ne * Nz * Nr * Nc && idy <= 2 * Ne * Nc * Nz * Nr

            # Pre-calculate values based on idx (target element)
            ri_col_index = idx # Assuming direct indexing for now, adjust if needed
            ri_val = ri_unique_row[ri_col_index]
            ri_sq = ri_val^2
            zi_val = zi[idx]

            # Pre-calculate values based on idy (source element)
            aj_val = aj[idy]
            bj_val = bj[idy]
            cj_val = cj[idy]
            dj_val = dj[idy]
            ej_val = (aj_val + bj_val) * 0.5

            # Pre-calculate differences involving z
            dzi_cj = zi_val - cj_val
            dzi_dj = zi_val - dj_val

            sum_K = 0.0
            sum_MBz = 0.0
            sum_MBr = 0.0

            # Determine the number of integration points
            if ((region1_start <= idx <= region1_end && region1_start <= idy <= region1_end) || (region2_start <= idx <= region2_end && region2_start <= idy <= region2_end))
                θ_range_K = θ_range_K_high
                dθ_K = dθ_K_high
                θ_range_M = θ_range_M_high
                dθ_M = dθ_M_high
            else
                θ_range_K = θ_range_K_low
                dθ_K = dθ_K_low
                θ_range_M = θ_range_M_low
                dθ_M = dθ_M_low
            end

            epsilon = 1.0e-10

            # Integration for K
            for k in 1:length(θ_range_K)
                θ = θ_range_K[k]
                cos_theta = cos(θ)

                # Reuse pre-calculated values
                term1_num_sq = dzi_cj^2 + ri_sq + aj_val^2 - 2.0*ri_val*aj_val*cos_theta
                term1_num = sqrt(term1_num_sq) + dzi_cj
                term1_den_sq = dzi_dj^2 + ri_sq + aj_val^2 - 2.0*ri_val*aj_val*cos_theta
                term1_den = sqrt(term1_den_sq) + dzi_dj
                safe_num1 = max(term1_num, epsilon)
                safe_den1 = max(term1_den, epsilon)
                log_ratio1 = log(max(min(safe_num1 / safe_den1, 1.0e30), 1.0e-30))
                term1 = log_ratio1 * aj_val * cos_theta

                term2_num_sq = dzi_cj^2 + ri_sq + ej_val^2 - 2.0*ri_val*ej_val*cos_theta
                term2_num = sqrt(term2_num_sq) + dzi_cj
                term2_den_sq = dzi_dj^2 + ri_sq + ej_val^2 - 2.0*ri_val*ej_val*cos_theta
                term2_den = sqrt(term2_den_sq) + dzi_dj
                safe_num2 = max(term2_num, epsilon)
                safe_den2 = max(term2_den, epsilon)
                log_ratio2 = log(max(min(safe_num2 / safe_den2, 1.0e30), 1.0e-30))
                term2 = 4.0 * log_ratio2 * ej_val * cos_theta

                term3_num_sq = dzi_cj^2 + ri_sq + bj_val^2 - 2.0*ri_val*bj_val*cos_theta
                term3_num = sqrt(term3_num_sq) + dzi_cj
                term3_den_sq = dzi_dj^2 + ri_sq + bj_val^2 - 2.0*ri_val*bj_val*cos_theta
                term3_den = sqrt(term3_den_sq) + dzi_dj
                safe_num3 = max(term3_num, epsilon)
                safe_den3 = max(term3_den, epsilon)
                log_ratio3 = log(max(min(safe_num3 / safe_den3, 1.0e30), 1.0e-30))
                term3 = log_ratio3 * bj_val * cos_theta

                sum_K += (bj_val - aj_val) / 6.0 * (term1 + term2 + term3) * dθ_K
            end
            K[idx, idy] = sum_K

            # Integration for MBz and MBr with safeguards
            for k in 1:length(θ_range_M)
                θ = θ_range_M[k]
                sin_theta = sin(θ)

                # Protected calculations for MBz
                # Term 1 of MBz
                denom1a = aj_val^2 - 2.0 * sin_theta * aj_val * ri_val + ri_sq
                denom1a = max(denom1a, epsilon)
                rad1a_sq = dzi_dj^2 + aj_val^2 + ri_sq - 2.0 * aj_val * ri_val * sin_theta
                rad1a = sqrt(max(rad1a_sq, epsilon))
                rad1b_sq = dzi_cj^2 + aj_val^2 + ri_sq - 2.0 * aj_val * ri_val * sin_theta
                rad1b = sqrt(max(rad1b_sq, epsilon))
                term1a = (aj_val * (aj_val - ri_val * sin_theta) * dzi_dj) / (denom1a * rad1a)
                term1b = (aj_val * (aj_val - ri_val * sin_theta) * dzi_cj) / (denom1a * rad1b)

                # Term 2 of MBz
                denom2a = ej_val^2 - 2.0 * sin_theta * ej_val * ri_val + ri_sq
                denom2a = max(denom2a, epsilon)
                rad2a_sq = dzi_dj^2 + ej_val^2 + ri_sq - 2.0 * ej_val * ri_val * sin_theta
                rad2a = sqrt(max(rad2a_sq, epsilon))
                rad2b_sq = dzi_cj^2 + ej_val^2 + ri_sq - 2.0 * ej_val * ri_val * sin_theta
                rad2b = sqrt(max(rad2b_sq, epsilon))
                term2a = (ej_val * (ej_val - ri_val * sin_theta) * dzi_dj) / (denom2a * rad2a)
                term2b = (ej_val * (ej_val - ri_val * sin_theta) * dzi_cj) / (denom2a * rad2b)

                # Term 3 of MBz
                denom3a = bj_val^2 - 2.0 * sin_theta * bj_val * ri_val + ri_sq
                denom3a = max(denom3a, epsilon)
                rad3a_sq = dzi_dj^2 + bj_val^2 + ri_sq - 2.0 * bj_val * ri_val * sin_theta
                rad3a = sqrt(max(rad3a_sq, epsilon))
                rad3b_sq = dzi_cj^2 + bj_val^2 + ri_sq - 2.0 * bj_val * ri_val * sin_theta
                rad3b = sqrt(max(rad3b_sq, epsilon))
                term3a = (bj_val * (bj_val - ri_val * sin_theta) * dzi_dj) / (denom3a * rad3a)
                term3b = (bj_val * (bj_val - ri_val * sin_theta) * dzi_cj) / (denom3a * rad3b)

                term_MBz = (bj_val - aj_val) / 6.0 * (
                    (term1a - term1b) +
                    4.0 * (term2a - term2b) +
                    (term3a - term3b)
                )
                sum_MBz += term_MBz * dθ_M

                # Protected calculations for MBr
                rad1a_mbr_sq = dzi_dj^2 + aj_val^2 + ri_sq - 2.0 * aj_val * ri_val * sin_theta
                rad1a_mbr = sqrt(max(rad1a_mbr_sq, epsilon))
                rad1b_mbr_sq = dzi_cj^2 + aj_val^2 + ri_sq - 2.0 * aj_val * ri_val * sin_theta
                rad1b_mbr = sqrt(max(rad1b_mbr_sq, epsilon))
                term1_MBr = (aj_val * sin_theta) / rad1a_mbr - (aj_val * sin_theta) / rad1b_mbr

                rad2a_mbr_sq = dzi_dj^2 + ej_val^2 + ri_sq - 2.0 * ej_val * ri_val * sin_theta
                rad2a_mbr = sqrt(max(rad2a_mbr_sq, epsilon))
                rad2b_mbr_sq = dzi_cj^2 + ej_val^2 + ri_sq - 2.0 * ej_val * ri_val * sin_theta
                rad2b_mbr = sqrt(max(rad2b_mbr_sq, epsilon))
                term2_MBr = 4.0 * ((ej_val * sin_theta) / rad2a_mbr - (ej_val * sin_theta) / rad2b_mbr)

                rad3a_mbr_sq = dzi_dj^2 + bj_val^2 + ri_sq - 2.0 * bj_val * ri_val * sin_theta
                rad3a_mbr = sqrt(max(rad3a_mbr_sq, epsilon))
                rad3b_mbr_sq = dzi_cj^2 + bj_val^2 + ri_sq - 2.0 * bj_val * ri_val * sin_theta
                rad3b_mbr = sqrt(max(rad3b_mbr_sq, epsilon))
                term3_MBr = (bj_val * sin_theta) / rad3a_mbr - (bj_val * sin_theta) / rad3b_mbr

                sum_MBr += (bj_val - aj_val) / 6.0 * (term1_MBr + term2_MBr + term3_MBr) * dθ_M
            end

            MBz[idx, idy] = sum_MBz
            MBr[idx, idy] = sum_MBr
        end
        return nothing
    end

    # Configure kernel launch parameters
    max_threads = 256  # Maximum threads per block
    threads_dim = min(32, ceil(Int, sqrt(max_threads)))
    threads_per_block = (threads_dim, threads_dim)
    blocks_per_grid = (ceil(Int, (Ne * Nz * Nr * Nc)/threads_dim), ceil(Int, (2 * Ne * Nc * Nz * Nr)/threads_dim)) # Modified grid

    # Launch kernel 
    CUDA.@sync begin
        @cuda threads=threads_per_block blocks=blocks_per_grid cuda_kernel!(
            K, MBr, MBz, θ_range_K_low, θ_range_M_low, θ_range_K_high, θ_range_M_high, ri_unique_row, zi, aj, bj, cj, dj, N, dθ_K_low, dθ_M_low, dθ_K_high, dθ_M_high, region1_start, region1_end, region2_start, region2_end, dr, r_ranges_gpu)
    end

    # Transfer results back to CPU and scale
    K_cpu = Array(K)
    MBr_cpu = Array(MBr) ./ 1e3
    MBz_cpu = Array(MBz) ./ 1e3

    # Perform hcubature operation on diagonal elements of K
    for i in 1:Ne * Nz * Nr * Nc # Modified loop bound
        K_cpu[i, i] = hcubature(r -> integrand_K_diag(r[1], r[2], geometry.r_middle[i], geometry.z_middle[i], geometry.r_low[i], geometry.r_high[i], geometry.z_low[i], geometry.z_high[i]), [geometry.r_low[i], 0.0], [geometry.r_high[i], π], rtol = 1e-6, atol= 1e-9)[1]
        MBr_cpu[i, i] = 0
        MBz_cpu[i, i] = 0
    end

    K_cpu = K_cpu ./ 1e6

    # Handle potential NaN or Inf values
    K_cpu[isnan.(K_cpu) .| isinf.(K_cpu)] .= 0.0
    MBr_cpu[isnan.(MBr_cpu) .| isinf.(MBr_cpu)] .= 0.0
    MBz_cpu[isnan.(MBz_cpu) .| isinf.(MBz_cpu)] .= 0.0

    return K_cpu, MBr_cpu, MBz_cpu
end