##############################################################################
#
# VortexParticle2D.jl
#
# Part of CVortex.jl
# Representation of a 2D vortex particle.
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
	Representation of a 2D vortex particle in CVortex

You do not need this to use the CVortex API.

coord is a particle's position.
vorticity is the particle's vorticity
volume is the volume of the particle. This is only important for 
viscous vortex particle strength exchange methods (not included in this
wrapper)
"""
struct VortexParticle2D
    coord :: Vec2f
    vorticity :: Float32
    volume :: Float32
end

function VortexParticle2D(coord::Vec2f, vort::Real)
    return VortexParticle2D(coord, vort, 0.0)
end

function VortexParticle2D(coord::Vector{<:Real}, vort::Real, vol::Real)
	@assert(length(coord)==2)
    return VortexParticle2D(Vec2f(coord), vort, vol)
end

function VortexParticle2D(coord::Vector{<:Real}, vort::Real)
	@assert(length(coord)==3)
    return VortexParticle2D(coord, vort, 0.0)
end

function particle_induced_velocity(
    inducing_particle_position :: Vector{<:Real},
    inducing_particle_vorticity :: Real,
	measurement_point :: Vector{<:Real},
	kernel :: RegularisationFunction,
	regularisation_radius :: Real)
	
	check_particle_definition_2D(inducing_particle_position, 
		inducing_particle_vorticity)
	convertable_to_Vec2f_vect(measurement_point, "measurement_points")
	convertable_to_F32(regularisation_radius, "regularisation_radius")
    
    inducing_particle = VortexParticle2D(
        inducing_particle_position, inducing_particle_vorticity, 0.0)
    mes_pnt = Vec2f(measurement_point)
	ret = Vec2f(0., 0.)
	#=
    CVTX_EXPORT bsv_V2f cvtx_P2D_S2S_vel(
        const cvtx_P2D *self,
        const bsv_V2f mes_point,
        const cvtx_VortFunc *kernel,
        float regularisation_radius);
	=#
	ret = ccall(
			("cvtx_P2D_S2S_vel", libcvortex), 
			Vec2f, 
			(Ref{VortexParticle2D}, Vec2f, Ref{RegularisationFunction}, Cfloat),
			inducing_particle, mes_pnt, kernel, regularisation_radius
			)
	return [ret.x, ret.y]
end

function particle_induced_velocity(
    inducing_particle_position :: Matrix{<:Real},
    inducing_particle_vorticity :: Vector{<:Real},
	measurement_point :: Vector{<:Real},
	kernel :: RegularisationFunction,
	regularisation_radius :: Real)
    
	check_particle_definition_2D(inducing_particle_position, 
		inducing_particle_vorticity)
	convertable_to_Vec2f_vect(measurement_point, "measurement_points")
	convertable_to_F32(regularisation_radius, "regularisation_radius")
	
	np = size(inducing_particle_position)[1]
    inducing_particles = map(
        i->VortexParticle2D(
            inducing_particle_position[i, :], 
            inducing_particle_vorticity[i], 0.0),
        1:np)
    mes_pnt = Vec2f(measurement_point)
	
	pargarr = Vector{Ptr{VortexParticle2D}}(undef, length(inducing_particles))
	for i = 1 : length(pargarr)
		pargarr[i] = Base.pointer(inducing_particles, i)
	end
	ret =Vec2f(0., 0.)
	#=
    CVTX_EXPORT bsv_V2f cvtx_P2D_M2S_vel(
        const cvtx_P2D **array_start,
        const int num_particles,
        const bsv_V2f mes_point,
        const cvtx_VortFunc *kernel,
        float regularisation_radius);
	=#		
	ret = ccall(
			("cvtx_P2D_M2S_vel", libcvortex), 
			Vec2f, 
			(Ref{Ptr{VortexParticle2D}}, Cint, Vec2f, 
				Ref{RegularisationFunction}, Cfloat),
			pargarr, np, mes_pnt, kernel,	regularisation_radius)
	return [ret.x, ret.y]
end

function particle_induced_velocity(
    inducing_particle_position :: Matrix{<:Real},
    inducing_particle_vorticity :: Vector{<:Real},
	measurement_points :: Matrix{<:Real},
	kernel :: RegularisationFunction,
	regularisation_radius :: Real)
    
	check_particle_definition_2D(inducing_particle_position, 
		inducing_particle_vorticity)
	convertable_to_Vec2f_vect(measurement_points, "measurement_points")
	convertable_to_F32(regularisation_radius, "regularisation_radius")
		
	np = size(inducing_particle_position)[1]
	ni = size(measurement_points)[1]
    inducing_particles = map(
        i->VortexParticle2D(
            inducing_particle_position[i, :], 
            inducing_particle_vorticity[i], 0.0),
        1:np)
    mes_pnt = map(i->Vec2f(measurement_points[i,:]), 1:ni)
	
	pargarr = Vector{Ptr{VortexParticle2D}}(undef, np)
	for i = 1 : length(pargarr)
		pargarr[i] = Base.pointer(inducing_particles, i)
	end
	ret = Vector{Vec2f}(undef, ni)
	#=
    CVTX_EXPORT void cvtx_P2D_M2M_vel(
        const cvtx_P2D **array_start,
        const int num_particles,
        const bsv_V2f *mes_start,
        const int num_mes,
        bsv_V2f *result_array,
        const cvtx_VortFunc *kernel,
        float regularisation_radius);
	=#	
	ccall(
		("cvtx_P2D_M2M_vel", libcvortex), 
		Cvoid, 
		(Ptr{Ptr{VortexParticle2D}}, Cint, Ptr{Vec2f}, 
			Cint, Ref{Vec2f}, Ref{RegularisationFunction}, Cfloat),
		pargarr, np, mes_pnt, ni, ret, kernel, regularisation_radius)
	return Matrix{Float32}(ret)
end

function particle_visc_induced_dvort(
    inducing_particle_position :: Vector{<:Real},
	inducing_particle_vorticity :: Real,
	inducing_particle_area :: Real,
    induced_particle_position :: Vector{<:Real},
	induced_particle_vorticity :: Real,
	induced_particle_area :: Real,
	kernel :: RegularisationFunction,
	regularisation_radius :: Real,
	kinematic_visc :: Real)
		
	check_particle_definition_2D(inducing_particle_position, 
		inducing_particle_vorticity, inducing_particle_area)
	check_particle_definition_2D(induced_particle_position, 
		induced_particle_vorticity, induced_particle_area)
	convertable_to_F32(regularisation_radius, "regularisation_radius")
	@assert(kernel.eta_2D!=Cvoid, "You cannot use this regularisation "*
		"function for viscous simulations. Consider Winckelmans or Gaussian.")
	
	np = size(inducing_particle_position)[1]
	inducing_particle = VortexParticle2D(
			inducing_particle_position, 
			inducing_particle_vorticity, inducing_particle_area)
	induced_particle = VortexParticle2D(
			inducing_particle_position, 
			inducing_particle_vorticity, induced_particle_area)

	#=
	CVTX_EXPORT float cvtx_P2D_S2S_visc_dvort(
		const cvtx_P2D * self,
		const cvtx_P2D * induced_particle,
		const cvtx_VortFunc * kernel,
		float regularisation_radius,
		float kinematic_visc);
	=#
	ret = ccall(
		("cvtx_P2D_S2S_visc_dvort", libcvortex), 
		Cfloat, 
		(Ptr{VortexParticle2D}, Ptr{VortexParticle2D}, 
			Ref{RegularisationFunction}, Cfloat, Cfloat),
		inducing_particle, induced_particle, kernel, 
			regularisation_radius, kinematic_visc
		)
	return ret
end

function particle_visc_induced_dvort(
    inducing_particle_position :: Matrix{<:Real},
	inducing_particle_vorticity :: Vector{<:Real},
	inducing_particle_area :: Vector{<:Real},
    induced_particle_position :: Vector{<:Real},
	induced_particle_vorticity :: Real,
	induced_particle_area :: Real,
	kernel :: RegularisationFunction,
	regularisation_radius :: Real,
	kinematic_visc :: Real)
		
	check_particle_definition_2D(inducing_particle_position, 
		inducing_particle_vorticity, inducing_particle_area)
	check_particle_definition_2D(induced_particle_position, 
		induced_particle_vorticity, induced_particle_area)
	convertable_to_F32(regularisation_radius, "regularisation_radius")
	@assert(kernel.eta_2D!=Cvoid, "You cannot use this regularisation "*
		"function for viscous simulations. Consider Winckelmans or Gaussian.")
	
	np = size(inducing_particle_position)[1]
	inducing_particles = map(
		i->VortexParticle2D(
			inducing_particle_position[i, :], 
			inducing_particle_vorticity[i], inducing_particle_area[i]),
		1:np)
	induced_particle = VortexParticle2D(
			inducing_particle_position, 
			inducing_particle_vorticity, induced_particle_area)

	pargarr = Vector{Ptr{VortexParticle2D}}(undef, length(inducing_particles))
	for i = 1 : length(pargarr)
		pargarr[i] = Base.pointer(inducing_particles, i)
	end
	#=
	CVTX_EXPORT float cvtx_P2D_M2S_visc_dvort(
		const cvtx_P2D **array_start,
		const int num_particles,
		const cvtx_P2D *induced_particle,
		const cvtx_VortFunc *kernel,
		float regularisation_radius,
		float kinematic_visc);
	=#
	ret = ccall(
		("cvtx_P2D_M2S_visc_dvort", libcvortex), 
		Cfloat, 
		(Ptr{Ptr{VortexParticle2D}}, Cint, Ptr{VortexParticle2D}, 
			Ref{RegularisationFunction}, Cfloat, Cfloat),
		pargarr, length(inducing_particles), induced_particle,
			kernel, regularisation_radius, kinematic_visc
		)
	return ret
end

function particle_visc_induced_dvort(
    inducing_particle_position :: Matrix{<:Real},
	inducing_particle_vorticity :: Vector{<:Real},
	inducing_particle_area :: Vector{<:Real},
    induced_particle_position :: Matrix{<:Real},
	induced_particle_vorticity :: Vector{<:Real},
	induced_particle_area :: Vector{<:Real},
	kernel :: RegularisationFunction,
	regularisation_radius :: Real,
	kinematic_visc :: Real)
		
	check_particle_definition_2D(inducing_particle_position, 
		inducing_particle_vorticity, inducing_particle_area)
	check_particle_definition_2D(induced_particle_position, 
		induced_particle_vorticity, induced_particle_area)
	convertable_to_F32(regularisation_radius, "regularisation_radius")
	@assert(kernel.eta_2D!=Cvoid, "You cannot use this regularisation "*
		"function for viscous simulations. Consider Winckelmans or Gaussian.")
	
	np = size(inducing_particle_position)[1]
	ni = size(induced_particle_position)[1]
	inducing_particles = map(
		i->VortexParticle2D(
			inducing_particle_position[i, :], 
			inducing_particle_vorticity[i], inducing_particle_area[i]),
		1:np)
	induced_particles = map(
		i->VortexParticle2D(
			inducing_particle_position[i, :], 
			inducing_particle_vorticity[i], induced_particle_area[i]),
		1:ni)

	pargarr = Vector{Ptr{VortexParticle2D}}(undef, length(inducing_particles))
	for i = 1 : length(pargarr)
		pargarr[i] = Base.pointer(inducing_particles, i)
	end
	indarg = Vector{Ptr{VortexParticle2D}}(undef, ni)
	for i = 1 : length(indarg)
		indarg[i] = Base.pointer(induced_particles, i)
	end
	ret = Vector{Float32}(undef, ni)
	#=
	CVTX_EXPORT void cvtx_P2D_M2M_visc_dvort(
		const cvtx_P2D **array_start,
		const int num_particles,
		const cvtx_P2D **induced_start,
		const int num_induced,
		float *result_array,
		const cvtx_VortFunc *kernel,
		float regularisation_radius,
		float kinematic_visc)
	=#
	ccall(
		("cvtx_P2D_M2M_visc_dvort", libcvortex), 
		Cvoid, 
		(Ptr{Ptr{VortexParticle2D}}, Cint, Ptr{Ptr{VortexParticle2D}}, Cint, 
			Ptr{Float32}, Ref{RegularisationFunction}, Cfloat, Cfloat),
		pargarr, length(inducing_particles), indarg, length(induced_particles),
			ret, kernel, regularisation_radius, kinematic_visc
		)
	return ret
end

function redistribute_particles_on_grid(
    particle_positions :: Matrix{<:Real},
	particle_vorticities :: Vector{<:Real},
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
	check_particle_definition_2D(particle_positions, 
		particle_vorticities)
	convertable_to_F32(grid_density, "grid_density")
	convertable_to_F32(negligible_vort, "negligible_vort")
		
	np = size(particle_positions)[1]
	inducing_particles = map(
		i->VortexParticle2D(
			particle_positions[i, :], 
			particle_vorticities[i], 0.0),
		1:np)
	
	pargarr = Vector{Ptr{VortexParticle2D}}(undef, np)
	for i = 1 : length(pargarr)
		pargarr[i] = Base.pointer(inducing_particles, i)
	end
	#=
	CVTX_EXPORT int cvtx_P2D_redistribute_on_grid(
		const cvtx_P2D **input_array_start,
		const int n_input_particles,
		cvtx_P2D *output_particles,		/* input is &(*cvtx_P2D) to write to */
		int max_output_particles,		/* Set to resultant num particles.   */
		const cvtx_RedistFunc *redistributor,
		float grid_density,
		float negligible_vort)
	=#
	if max_new_particles == -1
		max_new_particles = ccall(
			("cvtx_P2D_redistribute_on_grid", libcvortex), 
			Cint, 
			(Ptr{Ptr{VortexParticle2D}}, Cint, Ptr{VortexParticle2D}, 
				Cint, Ref{RedistributionFunction}, Cfloat, Cfloat),
			pargarr, np, C_NULL, 1, redistribution_function, 
			grid_density, negligible_vort)
	end

	ret = Vector{VortexParticle2D}(undef, max_new_particles)
	nnp = ccall(
		("cvtx_P2D_redistribute_on_grid", libcvortex), 
		Cint, 
		(Ptr{Ptr{VortexParticle2D}}, Cint, Ptr{VortexParticle2D}, 
			Cint, Ref{RedistributionFunction}, Cfloat, Cfloat),
		pargarr, np, ret, max_new_particles, redistribution_function, 
		grid_density, negligible_vort)

	# nnp is the number of new particles.
	nvorts = zeros(Float32, nnp)
	nposns = zeros(Float32, nnp, 2)
	nareas = zeros(Float32, nnp)
	for i = 1 : nnp
		nvorts[i] = ret[i].vorticity
		nposns[i, :] = Vector{Float32}(ret[i].coord)
		nareas[i] = ret[i].volume
	end
	return nposns, nvorts, nareas
end
