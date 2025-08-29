function CreationDphiMatrix2PancakeCoils(geometry, coil)
	# Definición matriz de superficie.
	s = geometry.s
	Nz = geometry.Nz
	Nr = geometry.Nr
	Nc = geometry.Nc
	Ne = geometry.Ne
	N = Nz*Nr*Nc*Ne

	S = zeros(N,N)
	for N_espira = 1:Ne*Nc # Tengo un doble pancake
		pos = (N_espira-1)*Nr*Nz+1:(N_espira)*Nr*Nz
		S[(N_espira-1)*Nz*Nr+1:(N_espira)*Nz*Nr, pos] = repeat(s[pos], 1, Nz*Nr)'
	end
	coil.Ss_super = S
    return S
end