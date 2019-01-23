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
		
const libcvortex = joinpath(dirname(dirname(@__FILE__)), "libcvortex")
function __init__()
	try
		dlopen(libcvortex)
	catch
        error("$(libcvortex) cannot be opened. Possible solutions:",
			"\n\tRebuild package and restart julia",
			"\n\tCheck that OpenCL is installed on your PC.\n")
	end
end

#--------------------------------------------------------------------------------------------
struct Vec3f
	x :: Float32
	y :: Float32
	z :: Float32
end

struct VortexFunc
	g_fn :: Ptr{Cvoid}			# Actually float(*g_fn)(float rho)
	zeta_fn :: Ptr{Cvoid}		# Actually float(*zeta_fn)(float rho)
	combined_fn :: Ptr{Cvoid}	# Actually void(*combined_fn)(float rho, float* g, float* zeta)
	eta_fn :: Ptr{Cvoid}		# Actually float(*eta_fn)(float rho)
	cl_kernel_name_ext :: NTuple{32, Cchar}	# Char[32]
end

struct VortexParticle
	coord :: Vec3f
	vorticity :: Vec3f
	volume :: Float32
end

#= Functions to to get VortexFunc structures =#
function VortFunc_singular()
	ret = ccall(("cvtx_VortFunc_singular", "cvortex"), VortexFunc, ())
	return ret;
end
function VortFunc_winckelmans()
	ret = ccall(("cvtx_VortFunc_winckelmans", "cvortex"), VortexFunc, ())
	return ret;
end
function VortFunc_planetary()
	ret = ccall(("cvtx_VortFunc_planetary", "cvortex"), VortexFunc, ())
	return ret;
end
function VortFunc_gaussian()
	ret = ccall(("cvtx_VortFunc_gaussian", "cvortex"), VortexFunc, ())
	return ret;
end

#= Functions to compute effects on a vortex particle =#
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
			("cvtx_Particle_ind_vel", "cvortex"), 
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
			("cvtx_ParticleArr_ind_vel", "cvortex"), 
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
		("cvtx_ParticleArr_Arr_ind_vel", "cvortex"), 
		Cvoid, 
		(Ptr{Ptr{VortexParticle}}, Cint, Ptr{Vec3f}, 
			Cint, Ref{Vec3f}, Ref{VortexFunc}, Cfloat),
		pargarr, length(inducing_particles), measurement_points, 
			length(measurement_points), ret, kernel, regularisation_radius
		)
	return ret
end

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
			("cvtx_Particle_ind_dvort", "cvortex"), 
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
			("cvtx_ParticleArr_ind_dvort", "cvortex"), 
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
		("cvtx_ParticleArr_Arr_ind_dvort", "cvortex"), 
		Cvoid, 
		(Ptr{Ptr{VortexParticle}}, Cint, Ptr{Ptr{VortexParticle}}, Cint, 
			Ptr{Vec3f}, Ref{VortexFunc}, Cfloat),
		pargarr, length(inducing_particles), indarg, length(induced_particles),
			ret, kernel, regularisation_radius
		)
	return ret
end

end #module
