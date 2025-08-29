function ApplySymmetry12PancakeCoils_cpu(K, MBr, MBz, coil)
	# Reducción de la matriz de campo debido a la simetría
	N_puntos_total=coil.Nc*coil.Nr*coil.Nz*coil.Ne

	# Aplicar al superconductor
	pos_total = zeros(N_puntos_total,1)
	it = 1
	for j=coil.Nr*coil.Ne:1:2*coil.Nr*coil.Ne-1
	    pos_total[(it-1)*coil.Nz*coil.Nc+1:it*coil.Nz*coil.Nc,:]=(coil.Nz*coil.Nc*(j+1):-1:1+j*coil.Nz*coil.Nc)'
		it += 1
	end

	Ks=zeros(size(K,1),N_puntos_total)
	MBzs=zeros(size(K,1),N_puntos_total)
	MBrs=zeros(size(K,1),N_puntos_total)
	
	# Simetría superconductor
	Ks[:,1:N_puntos_total].=K[:,1:N_puntos_total].+K[:,Int.(pos_total[:,1])]
	MBrs[:,1:N_puntos_total].=MBr[:,1:N_puntos_total].+MBr[:,Int.(pos_total[:,1])]
	MBzs[:,1:N_puntos_total].=MBz[:,1:N_puntos_total].+MBz[:,Int.(pos_total[:,1])]

    # Reducción matrices
	keep_rows = trues(size(Ks, 1))
    keep_rows[Int.(pos_total[:, 1])] .= false 
    Ks = Ks[keep_rows, :]
	MBrs = MBrs[keep_rows, :]
	MBzs = MBzs[keep_rows, :]

	coil.Ks = Ks
    coil.MBrs = MBrs
    coil.MBzs = MBzs

	return Ks, MBrs, MBzs
end