function BuildGeometry12PancakeCoils(geometry)
	# Si se está trabajando con hierro y material superconductor, 1º superconductor, 2º hierro
	
	geometry.r_low = geometry.r_low
	geometry.r_middle = geometry.r_middle
	geometry.r_high = geometry.r_high
	
	geometry.z_low = geometry.z_low
	geometry.z_middle = geometry.z_middle
	geometry.z_high = geometry.z_high
	
	return geometry
end