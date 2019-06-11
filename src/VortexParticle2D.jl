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
    inducing_particle_vorticity :: Float32,
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
