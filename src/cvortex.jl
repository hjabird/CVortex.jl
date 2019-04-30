##############################################################################
#
# cvortex.jl
#
# Provides GPU accelerated vortex particle and vortex filaments methods. A
# wrapper for the C cvortex library.
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
cvortex contains GPU accelerated method for vortex particles and vortex 
filaments.
"""
module cvortex

export 	Vec3f,
		VortexFunc,
		VortexParticle,
		VortFunc_singular,
		VortFunc_gaussian,
		VortFunc_winckelmans,
		VortFunc_planetary,
		induced_velocity,
		induced_dvort

import Libdl: dlopen
		
const libcvortex = joinpath(dirname(dirname(@__FILE__)), "deps/libcvortex")
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

#------------------------------------------------------------------------------
"""
	A 3D Float32 vector.
"""
struct Vec3f
	x :: Float32
	y :: Float32
	z :: Float32
end

"""
	A 3D vortex regularisation fuction

	The method by which the singular nature of a vortex particle is handelled.
	Generally, this structure is best obtained via VortFunc_*. For example
	VortFunc_singular(), VortFunc_planetary(), VortFunc_winckelmans() or
	VortFunc_gaussian().

	Exposure of this allows the use of user kernels in the CPU multithreaded
	implementations in cvortex. If you're interested in this, looking at the
	cvortex library and the source of cvortex.jl is suggested to understand
	the required functions signitures. 
"""
struct VortexFunc
	g_fn :: Ptr{Cvoid}			# Actually float(*g_fn)(float rho)
	zeta_fn :: Ptr{Cvoid}		# Actually float(*zeta_fn)(float rho)
	combined_fn :: Ptr{Cvoid}	# Actually void(*combined_fn)(float rho, float* g, float* zeta)
	eta_fn :: Ptr{Cvoid}		# Actually float(*eta_fn)(float rho)
	cl_kernel_name_ext :: NTuple{32, Cchar}	# Char[32]
end

"""
	Representation of a vortex particle.

	coord is a particle's position.
	vorticity is the particle's vorticity
	volume is the volume of the particle. This is only important for 
	viscous vortex particle strength exchange methods (not included in this
	wrapper)
"""
struct VortexParticle
	coord :: Vec3f
	vorticity :: Vec3f
	volume :: Float32
end

"""
	Representation of a straight singular vortex filament

	The filament starts at coord1, ends at coord2 and has vorticity per unit
	length of vorticity_density.
"""
struct VortexFilament
	coord1 :: Vec3f
	coord2 :: Vec3f
	vorticity_density :: Float32
end

#= Functions to to get VortexFunc structures =#
function VortFunc_singular()
	ret = ccall(("cvtx_VortFunc_singular", libcvortex), VortexFunc, ())
	return ret;
end
function VortFunc_winckelmans()
	ret = ccall(("cvtx_VortFunc_winckelmans", libcvortex), VortexFunc, ())
	return ret;
end
function VortFunc_planetary()
	ret = ccall(("cvtx_VortFunc_planetary", libcvortex), VortexFunc, ())
	return ret;
end
function VortFunc_gaussian()
	ret = ccall(("cvtx_VortFunc_gaussian", libcvortex), VortexFunc, ())
	return ret;
end

#= Functions to compute effects on a vortex particle =#
"""
	Compute the velocity induced in the flow field by inducers at measurement 
	points.

	Argument 1: inducer - The vortex particle(s) or filament(s) inducing the
		the velocity.
	Argument 2: measurement points.
	Additional arguments are required. See methods(induced_velocity) for a 
	list of methods.

	Vortex particles also require a regularisation kernel::VortexFunc and 
	regularisation radius.
"""
function induced_velocity(
	inducing_particle :: VortexParticle,
	measurement_point :: Vec3f,
	kernel :: VortexFunc,
	regularisation_radius :: T) where T <: Real
	
	#=
	cvtx_Vec3f cvtx_Particle_ind_vel(
		const cvtx_Particle *self, 
		const cvtx_Vec3f mes_point, 
		const cvtx_VortFunc *kernel,
		float regularisation_radius);
	=#
	ret = ccall(
			("cvtx_Particle_ind_vel", libcvortex), 
			Vec3f, 
			(Ref{VortexParticle}, Vec3f, Ref{VortexFunc}, Cfloat),
			inducing_particle, measurement_point, 
				kernel, regularisation_radius
			)
	return ret
end

function induced_velocity(
	inducing_particles :: Vector{VortexParticle},
	measurement_point :: Vec3f,
	kernel :: VortexFunc,
	regularisation_radius :: T) where T <: Real
	
	pargarr = Vector{Ptr{VortexParticle}}(undef, length(inducing_particles))
	for i = 1 : length(pargarr)
		pargarr[i] = Base.pointer(inducing_particles, i)
	end
	#=
	cvtx_Vec3f cvtx_ParticleArr_ind_vel(
		const cvtx_Particle **array_start,
		const int num_particles,
		const cvtx_Vec3f mes_point,
		const cvtx_VortFunc *kernel,
		float regularisation_radius);
	=#		
	ret = ccall(
			("cvtx_ParticleArr_ind_vel", libcvortex), 
			Vec3f, 
			(Ref{Ptr{VortexParticle}}, Cint, Vec3f, Ref{VortexFunc}, Cfloat),
			pargarr, length(inducing_particles), measurement_point, 
				kernel,	regularisation_radius
			)
	return ret
end

function induced_velocity(
	inducing_particles :: Vector{VortexParticle},
	measurement_points :: Vector{Vec3f},
	kernel :: VortexFunc,
	regularisation_radius :: T) where T <: Real
	
	
	pargarr = Vector{Ptr{VortexParticle}}(undef, length(inducing_particles))
	for i = 1 : length(pargarr)
		pargarr[i] = Base.pointer(inducing_particles, i)
	end
	ret = Vector{Vec3f}(undef, length(measurement_points))
	#=
	void cvtx_ParticleArr_Arr_ind_vel(
		const cvtx_Particle **array_start,
		const int num_particles,
		const cvtx_Vec3f *mes_start,
		const int num_mes,
		cvtx_Vec3f *result_array,
		const cvtx_VortFunc *kernel,
		float regularisation_radius);
	=#	
	ccall(
		("cvtx_ParticleArr_Arr_ind_vel", libcvortex), 
		Cvoid, 
		(Ptr{Ptr{VortexParticle}}, Cint, Ptr{Vec3f}, 
			Cint, Ref{Vec3f}, Ref{VortexFunc}, Cfloat),
		pargarr, length(inducing_particles), measurement_points, 
			length(measurement_points), ret, kernel, regularisation_radius
		)
	return ret
end

"""
	Rate of change of vorticity induced on vortex particles by element in the 
	flowfield.

	
	See methods(induced_velocity) for a list of methods.
"""
function induced_dvort(
	inducing_particle :: VortexParticle,
	induced_particle :: VortexParticle,
	kernel :: VortexFunc,
	regularisation_radius :: T)  where T <: Real
	
	#=
	cvtx_Vec3f cvtx_Particle_ind_dvort(
		const cvtx_Particle *self, 
		const cvtx_Particle *induced_particle,
		const cvtx_VortFunc *kernel,
		float regularisation_radius);
	=#
	ret = ccall(
			("cvtx_Particle_ind_dvort", libcvortex), 
			Vec3f, 
			(Ref{VortexParticle}, Ref{VortexParticle}, Ref{VortexFunc}, Cfloat),
			inducing_particle, induced_particle, kernel, regularisation_radius
			)
	return ret
end

function induced_dvort(
	inducing_particles :: Vector{VortexParticle},
	induced_particle :: VortexParticle,
	kernel :: VortexFunc,
	regularisation_radius :: T) where T <: Real
	
	pargarr = Vector{Ptr{VortexParticle}}(undef, length(inducing_particles))
	for i = 1 : length(pargarr)
		pargarr[i] = Base.pointer(inducing_particles, i)
	end
	#=
	cvtx_Vec3f cvtx_ParticleArr_ind_dvort(
		const cvtx_Particle **array_start,
		const int num_particles,
		const cvtx_Particle *induced_particle,
		const cvtx_VortFunc *kernel,
		float regularisation_radius)
	=#
	ret = ccall(
			("cvtx_ParticleArr_ind_dvort", libcvortex), 
			Vec3f, 
			(Ref{Ptr{VortexParticle}}, Cint, Ref{VortexParticle}, Ref{VortexFunc}, Cfloat),
			pargarr, length(inducing_particles), induced_particle, kernel, regularisation_radius
			)
	return ret
end

function induced_dvort(
	inducing_particles :: Vector{VortexParticle},
	induced_particles :: Vector{VortexParticle},
	kernel :: VortexFunc,
	regularisation_radius :: T) where T <: Real
	
	pargarr = Vector{Ptr{VortexParticle}}(undef, length(inducing_particles))
	for i = 1 : length(pargarr)
		pargarr[i] = Base.pointer(inducing_particles, i)
	end
	indarg = Vector{Ptr{VortexParticle}}(undef, length(induced_particles))
	for i = 1 : length(indarg)
		indarg[i] = Base.pointer(induced_particles, i)
	end
	ret = Vector{Vec3f}(undef, length(induced_particles))
	#=
	void cvtx_ParticleArr_Arr_ind_dvort(
		const cvtx_Particle **array_start,
		const int num_particles,
		const cvtx_Particle **induced_start,
		const int num_induced,
		cvtx_Vec3f *result_array,
		const cvtx_VortFunc *kernel,
		float regularisation_radius)
	=#
	ccall(
		("cvtx_ParticleArr_Arr_ind_dvort", libcvortex), 
		Cvoid, 
		(Ptr{Ptr{VortexParticle}}, Cint, Ptr{Ptr{VortexParticle}}, Cint, 
			Ptr{Vec3f}, Ref{VortexFunc}, Cfloat),
		pargarr, length(inducing_particles), indarg, length(induced_particles),
			ret, kernel, regularisation_radius
		)
	return ret
end

#= Functions to compute effects on a vortex filament =#
function induced_velocity(
	inducing_filaments :: VortexFilament,
	measurement_point :: Vec3f) where T <: Real
	
	#=
	bsv_V3f cvtx_StraightVortFil_ind_vel(
		const cvtx_StraightVortFil *self,
		const bsv_V3f mes_point);
	=#
	ret = ccall(
			("cvtx_StraightVortexFilament_ind_vel", libcvortex), 
			Vec3f, 
			(Ref{VortexFilament}, Vec3f),
			inducing_filaments, measurement_point
			)
	return ret
end

function induced_velocity(
	inducing_filaments :: Vector{VortexFilament},
	measurement_point :: Vec3f) where T <: Real
	
	pargarr = Vector{Ptr{VortexFilament}}(undef, length(inducing_filaments))
	for i = 1 : length(pargarr)
		pargarr[i] = Base.pointer(inducing_filaments, i)
	end
	#=
	bsv_V3f cvtx_StraightVortFilArr_ind_vel(
		const cvtx_StraightVortFil **array_start,
		const int num_filaments,
		const bsv_V3f mes_point)
	=#		
	ret = ccall(
			("cvtx_StraightVortFilArr_ind_vel", libcvortex), 
			Vec3f, 
			(Ref{Ptr{VortexFilament}}, Cint, Vec3f),
			pargarr, length(inducing_filaments), measurement_point
			)
	return ret
end

function induced_velocity(
	inducing_filaments :: Vector{VortexFilament},
	measurement_points :: Vector{Vec3f}) where T <: Real
	
	pargarr = Vector{Ptr{VortexFilament}}(undef, length(inducing_filaments))
	for i = 1 : length(pargarr)
		pargarr[i] = Base.pointer(inducing_filaments, i)
	end
	ret = Vector{Vec3f}(undef, length(measurement_points))
	#=
	void cvtx_StraightVortFilArr_Arr_ind_vel(
		const cvtx_StraightVortFil **array_start,
		const int num_filaments,
		const bsv_V3f *mes_start,
		const int num_mes,
		bsv_V3f *result_array)
	=#	
	ccall(
		("cvtx_StraightVortFilArr_Arr_ind_vel", libcvortex), 
		Cvoid, 
		(Ptr{Ptr{VortexFilament}}, Cint, Ptr{Vec3f}, 
			Cint, Ref{Vec3f}),
		pargarr, length(inducing_filaments), measurement_points, 
			length(measurement_points), ret
		)
	return ret
end

function induced_dvort(
	inducing_filaments :: VortexFilament,
	induced_particle :: VortexParticle)  where T <: Real
	
	#=
	bsv_V3f cvtx_StraightVortFil_ind_dvort(
		const cvtx_StraightVortFil *self,
		const cvtx_Particle *induced_particle);
	=#
	ret = ccall(
			("cvtx_StraightVortFil_ind_dvort", libcvortex), 
			Vec3f, 
			(Ref{VortexFilament}, Ref{VortexParticle}),
			inducing_filaments, induced_particle
			)
	return ret
end

function induced_dvort(
	inducing_filaments :: Vector{VortexFilament},
	induced_particle :: VortexParticle) where T <: Real
	
	pargarr = Vector{Ptr{VortexFilament}}(undef, length(inducing_filaments))
	for i = 1 : length(pargarr)
		pargarr[i] = Base.pointer(inducing_filaments, i)
	end
	#=
	bsv_V3f cvtx_StraightVortFilArr_ind_dvort(
		const cvtx_StraightVortFil **array_start,
		const int num_filaments,
		const cvtx_Particle *induced_particle);
	=#
	ret = ccall(
			("cvtx_StraightVortFilArr_ind_dvort", libcvortex), 
			Vec3f, 
			(Ref{Ptr{VortexFilament}}, Cint, Ref{VortexParticle}),
			pargarr, length(inducing_filaments), induced_particle
			)
	return ret
end

function induced_dvort(
	inducing_filaments :: Vector{VortexFilament},
	induced_particles :: Vector{VortexParticle}) where T <: Real
	
	pargarr = Vector{Ptr{VortexFilament}}(undef, length(inducing_filaments))
	for i = 1 : length(pargarr)
		pargarr[i] = Base.pointer(inducing_filaments, i)
	end
	indarg = Vector{Ptr{VortexParticle}}(undef, length(induced_particles))
	for i = 1 : length(indarg)
		indarg[i] = Base.pointer(induced_particles, i)
	end
	ret = Vector{Vec3f}(undef, length(induced_particles))
	#=
	void cvtx_StraightVortFilArr_Arr_ind_dvort(
		const cvtx_StraightVortFil **array_start,
		const int num_filaments,
		const cvtx_Particle **induced_start,
		const int num_induced,
		bsv_V3f *result_array);
	=#
	ccall(
		("cvtx_StraightVortFilArr_Arr_ind_dvort", libcvortex), 
		Cvoid, 
		(Ptr{Ptr{VortexFilament}}, Cint, Ptr{Ptr{VortexParticle}}, Cint, 
			Ptr{Vec3f}),
		pargarr, length(inducing_filaments), indarg, length(induced_particles),
			ret
		)
	return ret
end

"""
	The influence of vortex filaments on normal velocities at points in the
	domain.

	A list of i filaments induces a velocity at j measurement points with 
	j corresponding measurement directions. The velocity in these
	measurement directions is returned as a matrix of j by i.
"""
function induced_velocity_influence_matrix(
	inducing_filaments :: Vector{VortexFilament},
	measurement_points :: Vector{Vec3f},
	measurement_directions :: Vector{Vec3f}) where T <: Real
	
	pargarr = Vector{Ptr{VortexFilament}}(undef, length(inducing_filaments))
	for i = 1 : length(pargarr)
		pargarr[i] = Base.pointer(inducing_filaments, i)
	end
	# Julia is column major, C is row major. 
	ret = Matrix{Float32}(undef, length(inducing_filaments), 
		length(measurement_points))
	#=void cvtx_StraightVortFilArr_inf_mtrx(
		const cvtx_StraightVortFil **array_start,
		const int num_filaments,
		const bsv_V3f *mes_start,
		const bsv_V3f *dir_start,
		const int num_mes,
		float *result_matrix); 
	=#
	ccall(
		("cvtx_StraightVortFilArr_inf_mtrx", libcvortex), 
		Cvoid, 
		(Ptr{Ptr{VortexFilament}}, Cint, Ptr{Vec3f}, Ptr{Vec3f},
			Cint, Ref{Float32}),
		pargarr, length(inducing_filaments), measurement_points, 
		measurement_directions,	length(measurement_points), ret
		)
	return transpose(ret) # Fix row major -> column major
end

#= Ease of use functions ---------------------------------------------------=#
function Base.convert(::Vector{T}, a::Vec3f) where T<:Real
	return Vector{T}([a.x, a.y, a.z])
end

function Base.convert(::Matrix{T}, a::Vector{Vec3f}) where T <: Real
	len = length(a)
	mat = Matrix{T}(undef, len, 3)
	for i = 1 : len
		mat[i, :] = [a.x, a.y, a.z]
	end
	return mat
end

function Base.convert(::Vec3f, a::Vector{T}) where T<:Real
	@assert(length(a) == 3, "Vector is expected to be of length 3. "*
		"Actual length is ", length(a),".")
	return Vec3f(a[1], a[2], a[3])
end

function Base.convert(::Vector{Vec3f}, a::Matrix{T}) where T <: Real
	@assert(size(a)[2] == 3, "Input matrix is expected to be n rows by 3 "*
		"columns. Actual input matrix size was " * string(size(a)))
	len = size(a)[1]
	v = Vector{Vec3f}(undef, len)
	for i = 1 : len
		v[i] = convert(Vec3f, a[i, :])
	end
	return v
end

function induced_velocity(
	filaments :: Vector{VortexFilament},
	mes_points :: Matrix{T}) where T<:Real
	return induced_velocity(filaments, convert(Vector{Vec3f}, mes_points))
end

function induced_velocity(
	inducing_particles :: Vector{VortexParticle},
	measurement_points :: Matrix{T},
	kernel :: VortexFunc,
	regularisation_radius :: T) where T<:Real
	return induced_velocity(
		inducing_particles, 
		convert(Vector{Vec3f}, measurement_points),
		kernel,
		regularisation_radius)
end

#= cvortex accelerator controls --------------------------------------------=#
"""
	The number of GPU or other accelerators found by the cvortex library.

It is possible that accelerators may be listed multiple times if they can
be used by more than one installed platform.
To know how many are in use see cvortex_number_of_enabled_accelerators()
"""
function cvortex_number_of_accelerators()
	res = ccall(("cvtx_num_accelerators", libcvortex),
		Cint, ())
	return res
end

"""
	The number of accelerators that cvortex has been directed to use.

If no accelerators are in use this is zero, and the CPU is used for 
all computation.

Find which accelerators are enabled using cvortex_accelerator_enabled,
and enable and disable with cvortex_accelerator_enable and 
cvortex_accelerator_disable.
"""
function cvortex_number_of_enabled_accelerators()
	# int cvtx_num_enabled_accelerators();
	res = ccall(("cvtx_num_enabled_accelerators", libcvortex), Cint, ())
	return res
end

"""
	The name of an accelerator.

Input is an integer in the range 1:cvortex_number_of_accelerators(). Returns
the name of the accelerator as a string.
"""
function cvortex_accelerator_name(accelerator_id :: Int)
	@assert(accelerator_id >= 1, "Minimum accelerator id is 1.")
	@assert(accelerator_id <= cvortex_number_of_accelerators(),
		"accelerator_id is higher than the number of accelerators found.")
	# char* cvtx_accelerator_name(int accelerator_id);
	res = ccall(("cvtx_accelerator_name", libcvortex), 
		Cstring, (Cint,), accelerator_id-1)
	return unsafe_string(res)
end

"""
	States whether cvortex is in use.
"""
function cvortex_accelerator_enabled(accelerator_id :: Int)
	@assert(accelerator_id >= 1, "Minimum accelerator id is 1.")
	@assert(accelerator_id <= cvortex_number_of_accelerators(),
		"accelerator_id is higher than the number of accelerators found.")
	# int cvtx_accelerator_enabled(int accelerator_id);
	res = ccall(("cvtx_accelerator_enabled", libcvortex), 
		Cint, (Cint,), accelerator_id-1)
	return res
end

"""
	Allows cvortex to use an accelerator.
"""
function cvortex_accelerator_enable(accelerator_id :: Int)
	@assert(accelerator_id >= 1, "Minimum accelerator id is 1.")
	@assert(accelerator_id <= cvortex_number_of_accelerators(),
		"accelerator_id is higher than the number of accelerators found.")
	# void cvtx_accelerator_enable(int accelerator_id);
	ccall(("cvtx_accelerator_enable", libcvortex), 
		Cvoid, (Cint,), accelerator_id-1)
	return
end

"""
	Stops cvortex using an accelerator.
"""
function cvortex_accelerator_disable(accelerator_id :: Int)
	@assert(accelerator_id >= 1, "Minimum accelerator id is 1.")
	@assert(accelerator_id <= cvortex_number_of_accelerators(),
		"accelerator_id is higher than the number of accelerators found.")
	# void cvtx_accelerator_disable(int accelerator_id);
	ccall(("cvtx_accelerator_disable", libcvortex), 
		Cvoid, (Cint,), accelerator_id-1)
	return
end

end #module
