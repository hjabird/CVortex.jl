##############################################################################
#
# VortexParticle3D.jl
#
# Part of CVortex.jl
# Representation of a 3D vortex particle.
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
	Representation of a 3D vortex particle in CVortex

You do not need this to use the CVortex API.

coord is a particle's position.
vorticity is the particle's vorticity
volume is the volume of the particle. This is only important for 
viscous vortex particle strength exchange methods (not included in this
wrapper)
"""
struct VortexParticle3D
    coord :: Vec3f
    vorticity :: Vec3f
    volume :: Float32
end

function VortexParticle3D(coord::Vec3f, vort::Vec3f)
    return VortexParticle3D(coord, vort, 0.0)
end

function VortexParticle3D(coord::Vector{<:Real}, vort::Vector{<:Real}, vol::Real)
	@assert(length(coord)==3)
	@assert(length(vort)==3)
    return VortexParticle3D(Vec3f(coord), Vec3f(vort), vol)
end

function VortexParticle3D(coord::Vector{<:Real}, vort::Vector{<:Real})
	@assert(length(coord)==3)
	@assert(length(vort)==3)
    return VortexParticle3D(coord, vort, 0.0)
end

"""
# 3D vortex particle functions
Single particle -> single velocity measurement

	particle_induced_velocity(
		inducing_particle_position :: Vector{<:Real},
		inducing_particle_vorticity :: Vector{<:Real},
		measurement_point :: Vector{<:Real},
		kernel :: RegularisationFunction, regularisation_radius :: Real)

Multiple particles -> single velocity measurement

	particle_induced_velocity(
		inducing_particle_positions :: Matrix{<:Real},
		inducing_particle_vorticities :: Matrix{<:Real},
		measurement_point :: Vector{<:Real},
		kernel :: RegularisationFunction, regularisation_radius :: Real)

Multiple particles -> multiple velocity measurements

	particle_induced_velocity(
		inducing_particle_positions :: Matrix{<:Real},
		inducing_particle_vorticities :: Matrix{<:Real},
		measurement_points :: Matrix{<:Real},
		kernel :: RegularisationFunction, regularisation_radius :: Real)
		
# 2D vortex particle functions
Single particle -> single velocity measurement

	particle_induced_velocity(
		inducing_particle_position :: Vector{<:Real},
		inducing_particle_vorticity :: Real,
		measurement_point :: Vector{<:Real},
		kernel :: RegularisationFunction,
		regularisation_radius :: Real)

Multiple particles -> single velocity measurement

	particle_induced_velocity(
		inducing_particle_positions :: Matrix{<:Real},
		inducing_particle_vorticities :: Vector{<:Real},
		measurement_point :: Vector{<:Real},
		kernel :: RegularisationFunction,
		regularisation_radius :: Real)
	
Multiple particles -> multiple velocity measurements

	particle_induced_velocity(
		inducing_particle_positions :: Matrix{<:Real},
		inducing_particle_vorticities :: Vector{<:Real},
		measurement_points :: Matrix{<:Real},
		kernel :: RegularisationFunction,
		regularisation_radius :: Real)

Compute the velocity induced in the flow field by vortex particles. The
third (multiple-multiple) method may be GPU accelerated.

# Arguments
- `inducing_particle_position` : Position of inducing particles
- `inducing_particle_vorticity` : Vorticity of inducing particles
- `mesurement_points` : Measurement points 
- `kernel :: RegularisationFunction` : Regularisation function (winckelmans_regularisation() for example)
- `regularisation_radius :: Real` : Regularisation distance

In 3D, matrix arguments are expected to have size N by 3.
In 2D, matrix arguments are expected to have size N by 2.

The method will return velocity as:
- 3D - S2S: Vector length 3
- 3D - M2S: Vector length 3
- 3D - M2M: Matrix size N by 3
- 2D - S2S: Vector length 2
- 2D - M2S: Vector length 2
- 2D - M2M: Matrix size N by 2
"""
function particle_induced_velocity(
    inducing_particle_position :: Vector{<:Real},
    inducing_particle_vorticity :: Vector{<:Real},
	measurement_point :: Vector{<:Real},
	kernel :: RegularisationFunction,
	regularisation_radius :: Real)
	
	check_particle_definition_3D(inducing_particle_position, 
		inducing_particle_vorticity)
	convertable_to_Vec3f_vect(measurement_point, "measurement_points")
	convertable_to_F32(regularisation_radius, "regularisation_radius")
    
    inducing_particle = VortexParticle3D(
        inducing_particle_position, 
        inducing_particle_vorticity, 0.0)
    mes_pnt = Vec3f(measurement_point)
	ret = Vec3f(0., 0., 0.)
	#=
	CVTX_EXPORT bsv_V3f cvtx_P3D_S2S_vel(
		const cvtx_P3D *self,
		const bsv_V3f mes_point,
		const cvtx_VortFunc *kernel,
		float regularisation_radius);
	=#
	ret = ccall(
			("cvtx_P3D_S2S_vel", libcvortex), 
			Vec3f, 
			(Ref{VortexParticle3D}, Vec3f, Ref{RegularisationFunction}, Cfloat),
			inducing_particle, mes_pnt, kernel, regularisation_radius
			)
	return [ret.x, ret.y, ret.z]
end

function particle_induced_velocity(
    inducing_particle_position :: Matrix{<:Real},
    inducing_particle_vorticity :: Matrix{<:Real},
	measurement_point :: Vector{<:Real},
	kernel :: RegularisationFunction,
	regularisation_radius :: Real)
    
	check_particle_definition_3D(inducing_particle_position, 
		inducing_particle_vorticity)
	convertable_to_Vec3f_vect(measurement_point, "measurement_points")
	convertable_to_F32(regularisation_radius, "regularisation_radius")
	
	np = size(inducing_particle_position)[1]
    inducing_particles = map(
        i->VortexParticle3D(
            inducing_particle_position[i, :], 
            inducing_particle_vorticity[i, :], 0.0),
        1:np)
    mes_pnt = Vec3f(measurement_point)
	
	pargarr = Vector{Ptr{VortexParticle3D}}(undef, length(inducing_particles))
	for i = 1 : length(pargarr)
		pargarr[i] = Base.pointer(inducing_particles, i)
	end
	ret =Vec3f(0., 0., 0.)
	#=
	CVTX_EXPORT bsv_V3f cvtx_P3D_M2S_vel(
		const cvtx_P3D **array_start,
		const int num_particles,
		const bsv_V3f mes_point,
		const cvtx_VortFunc *kernel,
		float regularisation_radius);
	=#		
	ret = ccall(
			("cvtx_P3D_M2S_vel", libcvortex), 
			Vec3f, 
			(Ref{Ptr{VortexParticle3D}}, Cint, Vec3f, 
				Ref{RegularisationFunction}, Cfloat),
			pargarr, np, mes_pnt, kernel,	regularisation_radius)
	return [ret.x, ret.y, ret.z]
end

function particle_induced_velocity(
    inducing_particle_position :: Matrix{<:Real},
    inducing_particle_vorticity :: Matrix{<:Real},
	measurement_points :: Matrix{<:Real},
	kernel :: RegularisationFunction,
	regularisation_radius :: Real)
    
	check_particle_definition_3D(inducing_particle_position, 
		inducing_particle_vorticity)
	convertable_to_Vec3f_vect(measurement_points, "measurement_points")
	convertable_to_F32(regularisation_radius, "regularisation_radius")
		
	np = size(inducing_particle_position)[1]
	ni = size(measurement_points)[1]
    inducing_particles = map(
        i->VortexParticle3D(
            inducing_particle_position[i, :], 
            inducing_particle_vorticity[i, :], 0.0),
        1:np)
    mes_pnt = map(i->Vec3f(measurement_points[i,:]), 1:ni)
	
	pargarr = Vector{Ptr{VortexParticle3D}}(undef, np)
	for i = 1 : length(pargarr)
		pargarr[i] = Base.pointer(inducing_particles, i)
	end
	ret = Vector{Vec3f}(undef, ni)
	#=
	CVTX_EXPORT void cvtx_P3D_M2M_vel(
		const cvtx_P3D **array_start,
		const int num_particles,
		const bsv_V3f *mes_start,
		const int num_mes,
		bsv_V3f *result_array,
		const cvtx_VortFunc *kernel,
		float regularisation_radius);
	=#	
	ccall(
		("cvtx_P3D_M2M_vel", libcvortex), 
		Cvoid, 
		(Ptr{Ptr{VortexParticle3D}}, Cint, Ptr{Vec3f}, 
			Cint, Ref{Vec3f}, Ref{RegularisationFunction}, Cfloat),
		pargarr, np, mes_pnt, ni, ret, kernel, regularisation_radius)
	return Matrix{Float32}(ret)
end

"""

	particle_induced_dvort(
		inducing_particle_position :: Vector{<:Real},
		inducing_particle_vorticity :: Vector{<:Real},
		induced_particle_position :: Vector{<:Real},
		induced_particle_vorticity :: Vector{<:Real},
		kernel :: RegularisationFunction, regularisation_radius :: Real)

	particle_induced_dvort(
		inducing_particle_position :: Matrix{<:Real},
		inducing_particle_vorticity :: Matrix{<:Real},
		induced_particle_position :: Vector{<:Real},
		induced_particle_vorticity :: Vector{<:Real},
		kernel :: RegularisationFunction, regularisation_radius :: Real)

	particle_induced_dvort(
		inducing_particle_position :: Matrix{<:Real},
		inducing_particle_vorticity :: Matrix{<:Real},
		induced_particle_position :: Matrix{<:Real},
		induced_particle_vorticity :: Matrix{<:Real},
		kernel :: RegularisationFunction, regularisation_radius :: Real)

Rate of change of vorticity induced on vortex particles by vortex stretching. 
The third multiple-multiple variant may be GPU accelerated.
In 2D, vortex stretching does not occur, so there is no 2D counterpart.

# Arguments
- `inducing_particle_position` : Position of inducing particles
- `inducing_particle_vorticity` : Vorticity of inducing particles
- `induced_particle_position` : Position of induced particles
- `induced_particle_vorticity` : Vorticity of induced particles
- `kernel :: RegularisationFunction` : Regularisation function
(VortFunc_winckelmans for example)
- `regularisation_radius :: Real` : Regularisation distance

Vector arguments are expected to have length 3. Matrix arguments are
expected to have size N by 3.
"""
function particle_induced_dvort(
    inducing_particle_position :: Vector{<:Real},
    inducing_particle_vorticity :: Vector{<:Real},
    induced_particle_position :: Vector{<:Real},
    induced_particle_vorticity :: Vector{<:Real},
	kernel :: RegularisationFunction,
	regularisation_radius :: Real)
	
	check_particle_definition_3D(inducing_particle_position, 
		inducing_particle_vorticity)
	check_particle_definition_3D(induced_particle_position, 
		induced_particle_vorticity)
	convertable_to_F32(regularisation_radius, "regularisation_radius")
	
    inducing_particle = VortexParticle3D(
        inducing_particle_position, 
        inducing_particle_vorticity, 0.0)
	induced_particle = VortexParticle3D(
		induced_particle_position, 
		induced_particle_vorticity, 0.0)
	ret = Vec3f(0., 0., 0.)
	#=
	CVTX_EXPORT bsv_V3f cvtx_P3D_S2S_dvort(
		const cvtx_P3D *self,
		const cvtx_P3D *induced_particle,
		const cvtx_VortFunc *kernel,
		float regularisation_radius);
	=#
	ret = ccall(
			("cvtx_P3D_S2S_dvort", libcvortex), 
			Vec3f, 
			(Ref{VortexParticle3D}, Ref{VortexParticle3D}, 
				Ref{RegularisationFunction}, Cfloat),
			inducing_particle, induced_particle, kernel, regularisation_radius
			)
	return ret
end

function particle_induced_dvort(
    inducing_particle_position :: Matrix{<:Real},
    inducing_particle_vorticity :: Matrix{<:Real},
    induced_particle_position :: Vector{<:Real},
    induced_particle_vorticity :: Vector{<:Real},
	kernel :: RegularisationFunction,
	regularisation_radius :: Real)
		
	check_particle_definition_3D(inducing_particle_position, 
		inducing_particle_vorticity)
	check_particle_definition_3D(induced_particle_position, 
		induced_particle_vorticity)
	convertable_to_F32(regularisation_radius, "regularisation_radius")
	
	np = size(induced_particle_position)[1]
	inducing_particles = map(
		i->VortexParticle3D(
			inducing_particle_position[i, :], 
			inducing_particle_vorticity[i, :], 0.0),
		1:np)
	induced_particle = VortexParticle3D(
		induced_particle_position, 
		induced_particle_vorticity, 0.0)

	
	pargarr = Vector{Ptr{VortexParticle3D}}(undef, np)
	for i = 1 : length(pargarr)
		pargarr[i] = Base.pointer(inducing_particles, i)
	end
	ret = Vec3f(0., 0., 0.)
	#=
	CVTX_EXPORT bsv_V3f cvtx_P3D_M2S_dvort(
		const cvtx_P3D **array_start,
		const int num_particles,
		const cvtx_P3D *induced_particle,
		const cvtx_VortFunc *kernel,
		float regularisation_radius);
	=#
	ret = ccall(
			("cvtx_P3D_M2S_dvort", libcvortex), 
			Vec3f, 
			(Ref{Ptr{VortexParticle3D}}, Cint, Ref{VortexParticle3D}, 
				Ref{RegularisationFunction}, Cfloat),
			pargarr, np, induced_particle, kernel, regularisation_radius
			)
	return Vector{Float32}(ret)
end

function particle_induced_dvort(
    inducing_particle_position :: Matrix{<:Real},
    inducing_particle_vorticity :: Matrix{<:Real},
    induced_particle_position :: Matrix{<:Real},
    induced_particle_vorticity :: Matrix{<:Real},
	kernel :: RegularisationFunction,
	regularisation_radius :: Real)
		
	check_particle_definition_3D(inducing_particle_position, 
		inducing_particle_vorticity)
	check_particle_definition_3D(induced_particle_position, 
		induced_particle_vorticity)
	convertable_to_F32(regularisation_radius, "regularisation_radius")
	
	np = size(inducing_particle_position)[1]
	ni = size(induced_particle_position)[1]
	inducing_particles = map(
		i->VortexParticle3D(
			inducing_particle_position[i, :], 
			inducing_particle_vorticity[i, :], 0.0),
		1:np)
	induced_particles = map(
		i->VortexParticle3D(
			inducing_particle_position[i, :], 
			inducing_particle_vorticity[i, :], 0.0),
		1:ni)

	pargarr = Vector{Ptr{VortexParticle3D}}(undef, length(inducing_particles))
	for i = 1 : length(pargarr)
		pargarr[i] = Base.pointer(inducing_particles, i)
	end
	indarg = Vector{Ptr{VortexParticle3D}}(undef, ni)
	for i = 1 : length(indarg)
		indarg[i] = Base.pointer(induced_particles, i)
	end
	ret = Vector{Vec3f}(undef, ni)
	#=
	CVTX_EXPORT void cvtx_P3D_M2M_dvort(
		const cvtx_P3D **array_start,
		const int num_particles,
		const cvtx_P3D **induced_start,
		const int num_induced,
		bsv_V3f *result_array,
		const cvtx_VortFunc *kernel,
		float regularisation_radius);
	=#
	ccall(
		("cvtx_P3D_M2M_dvort", libcvortex), 
		Cvoid, 
		(Ptr{Ptr{VortexParticle3D}}, Cint, Ptr{Ptr{VortexParticle3D}}, Cint, 
			Ptr{Vec3f}, Ref{RegularisationFunction}, Cfloat),
		pargarr, length(inducing_particles), indarg, length(induced_particles),
			ret, kernel, regularisation_radius
		)
	return Matrix{Float32}(ret)
end

"""
# 3D functions

	particle_visc_induced_dvort(
		inducing_particle_position :: Vector{<:Real},
		inducing_particle_vorticity :: Vector{<:Real},
		inducing_particle_volume :: Real,
		induced_particle_position :: Vector{<:Real},
		induced_particle_vorticity :: Vector{<:Real},
		induced_particle_volume :: Real,
		kernel :: RegularisationFunction,
		regularisation_radius :: Real,
		kinematic_visc :: Real)
		
	particle_visc_induced_dvort(
		inducing_particle_position :: Matrix{<:Real},
		inducing_particle_vorticity :: Matrix{<:Real},
		inducing_particle_volume :: Vector{<:Real},
		induced_particle_position :: Vector{<:Real},
		induced_particle_vorticity :: Vector{<:Real},
		induced_particle_volume :: Real,
		kernel :: RegularisationFunction,
		regularisation_radius :: Real,
		kinematic_visc :: Real)
		
	particle_visc_induced_dvort(
		inducing_particle_position :: Matrix{<:Real},
		inducing_particle_vorticity :: Matrix{<:Real},
		inducing_particle_volume :: Vector{<:Real},
		induced_particle_position :: Matrix{<:Real},
		induced_particle_vorticity :: Matrix{<:Real},
		induced_particle_volume :: Vector{<:Real},
		kernel :: RegularisationFunction,
		regularisation_radius :: Real,
		kinematic_visc :: Real)
		
# 2D functions 

	particle_visc_induced_dvort(
		inducing_particle_position :: Vector{<:Real},
		inducing_particle_vorticity :: Real,
		inducing_particle_area :: Real,
		induced_particle_position :: Vector{<:Real},
		induced_particle_vorticity :: Real,
		induced_particle_area :: Real,
		kernel :: RegularisationFunction,
		regularisation_radius :: Real,
		kinematic_visc :: Real)

	particle_visc_induced_dvort(
		inducing_particle_position :: Matrix{<:Real},
		inducing_particle_vorticity :: Vector{<:Real},
		inducing_particle_area :: Vector{<:Real},
		induced_particle_position :: Vector{<:Real},
		induced_particle_vorticity :: Real,
		induced_particle_area :: Real,
		kernel :: RegularisationFunction,
		regularisation_radius :: Real,
		kinematic_visc :: Real)
		
	particle_visc_induced_dvort(
		inducing_particle_position :: Matrix{<:Real},
		inducing_particle_vorticity :: Vector{<:Real},
		inducing_particle_area :: Vector{<:Real},
		induced_particle_position :: Matrix{<:Real},
		induced_particle_vorticity :: Vector{<:Real},
		induced_particle_area :: Vector{<:Real},
		kernel :: RegularisationFunction,
		regularisation_radius :: Real,
		kinematic_visc :: Real)

The rate of change of vorticity induced on vortex particles by elements in the 
flowfield due to viscosity. 
The third multiple-multiple variant may be GPU accelerated.

# Arguments
- `inducing_particle_position` : Position of inducing particles
- `inducing_particle_vorticity` : Vorticity of inducing particles
- `inducing_particle_volume` : Volume of inducing particles
- `induced_particle_position` : Position of induced particles
- `induced_particle_vorticity` : Vorticity of induced particles
- `induced_particle_volume` : Volume of induced particles
- `kernel :: RegularisationFunction` : Regularisation function (winckelmans_regularisation() for example)
- `regularisation_radius :: Real` : Regularisation distance
- `kinematic_visc :: Real` : Kinematic viscosity
"""
function particle_visc_induced_dvort(
    inducing_particle_position :: Vector{<:Real},
	inducing_particle_vorticity :: Vector{<:Real},
	inducing_particle_volume :: Real,
    induced_particle_position :: Vector{<:Real},
	induced_particle_vorticity :: Vector{<:Real},
	induced_particle_volume :: Real,
	kernel :: RegularisationFunction,
	regularisation_radius :: Real,
	kinematic_visc :: Real)
	
	check_particle_definition_3D(inducing_particle_position, 
		inducing_particle_vorticity, inducing_particle_volume)
	check_particle_definition_3D(induced_particle_position, 
		induced_particle_vorticity, induced_particle_volume)
	convertable_to_F32(regularisation_radius, "regularisation_radius")
	@assert(kernel.eta_fn!=Cvoid, "You cannot use this regularisation "*
		"function for viscous simulations. Consider Winckelmans or Gaussian.")
	
    inducing_particle = VortexParticle3D(
        inducing_particle_position, 
        inducing_particle_vorticity, induced_particle_volume)
	induced_particle = VortexParticle3D(
		induced_particle_position, 
		induced_particle_vorticity, induced_particle_volume)
	ret = Vec3f(0., 0., 0.)
	#=
	CVTX_EXPORT bsv_V3f cvtx_P3D_S2S_visc_dvort(
		const cvtx_P3D *self,
		const cvtx_P3D *induced_particle,
		const cvtx_VortFunc *kernel,
		float regularisation_radius,
		float kinematic_visc);
	=#
	ret = ccall(
			("cvtx_P3D_S2S_visc_dvort", libcvortex), 
			Vec3f, 
			(Ref{VortexParticle3D}, Ref{VortexParticle3D}, 
				Ref{RegularisationFunction}, Cfloat, Cfloat),
			inducing_particle, induced_particle, kernel, 
				regularisation_radius, kinematic_visc
			)
	return ret
end

function particle_visc_induced_dvort(
    inducing_particle_position :: Matrix{<:Real},
	inducing_particle_vorticity :: Matrix{<:Real},
	inducing_particle_volume :: Vector{<:Real},
    induced_particle_position :: Vector{<:Real},
	induced_particle_vorticity :: Vector{<:Real},
	induced_particle_volume :: Real,
	kernel :: RegularisationFunction,
	regularisation_radius :: Real,
	kinematic_visc :: Real)
		
	check_particle_definition_3D(inducing_particle_position, 
		inducing_particle_vorticity, inducing_particle_volume)
	check_particle_definition_3D(induced_particle_position, 
		induced_particle_vorticity, induced_particle_volume)
	convertable_to_F32(regularisation_radius, "regularisation_radius")
	@assert(kernel.eta_fn!=Cvoid, "You cannot use this regularisation "*
		"function for viscous simulations. Consider Winckelmans or Gaussian.")
	
	np = size(induced_particle_position)[1]
	inducing_particles = map(
		i->VortexParticle3D(
			inducing_particle_position[i, :], 
			inducing_particle_vorticity[i, :], inducing_particle_volume[i]),
		1:np)
	induced_particle = VortexParticle3D(
		induced_particle_position, 
		induced_particle_vorticity, induced_particle_volume)

	pargarr = Vector{Ptr{VortexParticle3D}}(undef, np)
	for i = 1 : length(pargarr)
		pargarr[i] = Base.pointer(inducing_particles, i)
	end
	ret = Vec3f(0., 0., 0.)
	#=
	CVTX_EXPORT bsv_V3f cvtx_P3D_M2S_visc_dvort(
		const cvtx_P3D **array_start,
		const int num_particles,
		const cvtx_P3D *induced_particle,
		const cvtx_VortFunc *kernel,
		float regularisation_radius,
		float kinematic_visc);
	=#
	ret = ccall(
			("cvtx_P3D_M2S_visc_dvort", libcvortex), 
			Vec3f, 
			(Ref{Ptr{VortexParticle3D}}, Cint, Ref{VortexParticle3D}, 
				Ref{RegularisationFunction}, Cfloat, Cfloat),
			pargarr, np, induced_particle, kernel, 
				regularisation_radius, kinematic_visc
			)
	return Vector{Float32}(ret)
end

function particle_visc_induced_dvort(
    inducing_particle_position :: Matrix{<:Real},
	inducing_particle_vorticity :: Matrix{<:Real},
	inducing_particle_volume :: Vector{<:Real},
    induced_particle_position :: Matrix{<:Real},
	induced_particle_vorticity :: Matrix{<:Real},
	induced_particle_volume :: Vector{<:Real},
	kernel :: RegularisationFunction,
	regularisation_radius :: Real,
	kinematic_visc :: Real)
		
	check_particle_definition_3D(inducing_particle_position, 
		inducing_particle_vorticity, inducing_particle_volume)
	check_particle_definition_3D(induced_particle_position, 
		induced_particle_vorticity, induced_particle_volume)
	convertable_to_F32(regularisation_radius, "regularisation_radius")
	@assert(kernel.eta_3D!=Cvoid, "You cannot use this regularisation "*
		"function for viscous simulations. Consider Winckelmans or Gaussian.")
	
	np = size(inducing_particle_position)[1]
	ni = size(induced_particle_position)[1]
	inducing_particles = map(
		i->VortexParticle3D(
			inducing_particle_position[i, :], 
			inducing_particle_vorticity[i, :], inducing_particle_volume[i]),
		1:np)
	induced_particles = map(
		i->VortexParticle3D(
			inducing_particle_position[i, :], 
			inducing_particle_vorticity[i, :], induced_particle_volume[i]),
		1:ni)

	pargarr = Vector{Ptr{VortexParticle3D}}(undef, length(inducing_particles))
	for i = 1 : length(pargarr)
		pargarr[i] = Base.pointer(inducing_particles, i)
	end
	indarg = Vector{Ptr{VortexParticle3D}}(undef, ni)
	for i = 1 : length(indarg)
		indarg[i] = Base.pointer(induced_particles, i)
	end
	ret = Vector{Vec3f}(undef, ni)
	#=
	CVTX_EXPORT void cvtx_P3D_M2M_visc_dvort(
		const cvtx_P3D **array_start,
		const int num_particles,
		const cvtx_P3D **induced_start,
		const int num_induced,
		bsv_V3f *result_array,
		const cvtx_VortFunc *kernel,
		float regularisation_radius,
		float kinematic_visc);
	=#
	ccall(
		("cvtx_P3D_M2M_visc_dvort", libcvortex), 
		Cvoid, 
		(Ptr{Ptr{VortexParticle3D}}, Cint, Ptr{Ptr{VortexParticle3D}}, Cint, 
			Ptr{Vec3f}, Ref{RegularisationFunction}, Cfloat, Cfloat),
		pargarr, length(inducing_particles), indarg, length(induced_particles),
			ret, kernel, regularisation_radius, kinematic_visc
		)
	return Matrix{Float32}(ret)
end

"""
	redistribute_particles_on_grid(
		particle_positions :: Matrix{<:Real},
		particle_vorticities :: Matrix{<:Real},
		redistribution_function :: RedistributionFunction,
		grid_density :: Real;
		negligible_vort::Real=1e-4,
		max_new_particles::Integer=-1)

	redistribute_particles_on_grid(
		particle_positions :: Matrix{<:Real},
		inducing_particle_vorticity :: Vector{<:Real},
		redistribution_function :: RedistributionFunction,
		grid_density :: Real;
		negligible_vort::Real=1e-4,
		max_new_particles::Integer=-1)

Redistribute vorticity from vortex particles onto a regular grid.

# Arguments
- `particle_positions` : Position of particles to be redistributed.
- `particle_vorticities` : Vorticity of particles to be redistributed
- `redistribution_function :: RedistributionFunction` : Redistribution function (lambda3_redistribution() for example).
- `grid density :: Real` : The distance between newly placed particles on the grid.
- `negligible_vort :: Real` : The vorticity threshold at which to discard vortex particles. Given as abs(vorticity) as a proportion of the average particle's abs(vorticity)
- `max_new_particles::Integer` : The maximum number of particles that the grid can contain. -1 indicates that any number of particles can be created.

returns 

	particle_positions, particle_vorts, particle_areas = redistribute_particles_on_grid(...)
	
where the
number of created particles may be less that the max_new_particles variable, even
when the number of particles is limited by the max_new_particles variable.
"""
function redistribute_particles_on_grid(
    particle_positions :: Matrix{<:Real},
	particle_vorticities :: Matrix{<:Real},
	redistribution_function :: RedistributionFunction,
	grid_density :: Real;
	negligible_vort::Real=1e-4,
	max_new_particles::Integer=-1)

	@assert(0 <= negligible_vort < 1, "The negligible_vort vort must be"*
		" in [0, 1). Given "*string(negligible_vort)*"." )
	@assert((max_new_particles==-1)||(0<=max_new_particles), 
		"max_new_particles must be -1 (indicating any number of new particles"*
		") or a positive integer.")
	@assert(0<grid_density, "Grid density must be positive. Was "*
		string(grid_density)*".")
	check_particle_definition_3D(particle_positions, 
		particle_vorticities)
	convertable_to_F32(grid_density, "grid_density")
	convertable_to_F32(negligible_vort, "negligible_vort")
		
	np = size(particle_positions)[1]
	inducing_particles = map(
		i->VortexParticle3D(
			particle_positions[i, :], 
			particle_vorticities[i, :], 0.0),
		1:np)
	
	pargarr = Vector{Ptr{VortexParticle3D}}(undef, np)
	for i = 1 : length(pargarr)
		pargarr[i] = Base.pointer(inducing_particles, i)
	end
	#=
	CVTX_EXPORT int cvtx_P3D_redistribute_on_grid(
		const cvtx_P3D **input_array_start,
		const int n_input_particles,
		cvtx_P3D *output_particles,		/* input is &(*cvtx_P3D) to write to */
		int max_output_particles,		/* Set to resultant num particles.   */
		const cvtx_RedistFunc *redistributor,
		float grid_density,
		float negligible_vort);
	=#
	println(Base.pointer(pargarr, 1))
	println(Base.pointer(inducing_particles, 1))
	if max_new_particles == -1
		max_new_particles = ccall(
			("cvtx_P3D_redistribute_on_grid", libcvortex), 
			Cint, 
			(Ptr{Ptr{VortexParticle3D}}, Cint, Ptr{VortexParticle3D}, 
				Cint, Ref{RedistributionFunction}, Cfloat, Cfloat),
			pargarr, np, C_NULL, 1, redistribution_function, 
			grid_density, negligible_vort)
	end

	ret = Vector{VortexParticle3D}(undef, max_new_particles)
	nnp = ccall(
		("cvtx_P3D_redistribute_on_grid", libcvortex), 
		Cint, 
		(Ptr{Ptr{VortexParticle3D}}, Cint, Ptr{VortexParticle3D}, 
			Cint, Ref{RedistributionFunction}, Cfloat, Cfloat),
		pargarr, np, ret, max_new_particles, redistribution_function, 
		grid_density, negligible_vort)

	# nnp is the number of new particles.
	nvorts = zeros(Float32, nnp, 3)
	nposns = zeros(Float32, nnp, 3)
	nareas = zeros(Float32, nnp)
	for i = 1 : nnp
		nvorts[i, :] = Vector{Float32}(ret[i].vorticity)
		nposns[i, :] = Vector{Float32}(ret[i].coord)
		nareas[i] = ret[i].volume
	end
	return nposns, nvorts, nareas
end
