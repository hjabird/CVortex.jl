##############################################################################
#
# CVortex.jl
#
# Provides GPU accelerated vortex particle and vortex filaments methods. A
# wrapper for the C CVortex library.
#
# Copyright 2019 HJA Bird
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to 
# deal in the Software without restriction, including without limitation the 
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
##############################################################################

"""
	CVortex 
	
A GPU accelerated vortex particle and vortex filament library.
"""
module CVortex

	export 	particle_induced_velocity,
			filament_induced_velocity,
			particle_induced_dvort,
			filament_induced_dvort,
			particle_visc_induced_dvort,
			filament_induced_velocity_influence_matrix,
			RegularisationFunction,
			singular_regularisation,
			planetary_regularisation,
			gaussian_regularisation,
			winckelmans_regularisation,
			redistribute_particles_on_grid,
			RedistributionFunction,
			lambda0_redistribution,
			lambda1_redistribution,
			lambda2_redistribution,
			lambda3_redistribution,
			m4p_redistribution,
			number_of_accelerators,
			number_of_enabled_accelerators,
			accelerator_name,
			accelerator_enabled,
			accelerator_enable,
			accelerator_disable
	#-------------------------------------------------------------------------
	# Loading shared library binary (cvortex)

	import Libdl: dlopen

	const libcvortex = joinpath(
		dirname(dirname(@__FILE__)), "deps/libcvortex")
	function __init__()
		try
			dlopen(libcvortex)
		catch
			error("$(libcvortex) cannot be opened. Possible solutions:",
				"\n\tRebuild package and restart julia",
				"\n\tCheck that OpenCL is installed on your PC.\n")
		end
		# Inialisation is automatic:
		ccall(
			("cvtx_initialise", libcvortex),
			Cvoid, ())
	end

	#-------------------------------------------------------------------------
	include("ConvertionChecks.jl")
	include("Vec3f.jl")
	include("Vec2f.jl")
	include("Accelerators.jl")
	include("RegularisationFunction.jl")
	include("RedistributionFunction.jl")
	include("VortexParticle3D.jl")
	include("VortexParticle2D.jl")
	include("VortexFilament.jl")

end #module
