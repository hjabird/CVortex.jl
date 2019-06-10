##############################################################################
#
# VortexFilament.jl
#
# Part of CVortex.jl
# Representation of a vortex filament.
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
CVortex internal representation of a straight singular vortex filament

The filament starts at coord1, ends at coord2 and has vorticity per unit
length of vorticity_density.

You only need to use this if you plan on calling CVortex's underlying 
library directly.
"""
struct VortexFilament
    coord1 :: Vec3f
    coord2 :: Vec3f
    vorticity_density :: Float32
end

function VortexFilament(
    coord1::Vector{<:Real}, coord2::Vector{<:Real}, vort::Real)
    return VortexFilament(Vec3f(coord1), Vec3f(coord2), Float32(vort))
end

"""
    filament_induced_velocity(
        filament_start_coord :: Vector{<:Real},
        filament_end_coord :: Vector{<:Real},
        filament_strength :: Real,
        measurement_point :: Vector{<:Real})

    filament_induced_velocity(
        filament_start_coords :: Matrix{<:Real},
        filament_end_coords :: Matrix{<:Real},
        filament_strengths :: Vector{<:Real},
        measurement_point :: Vector{<:Real})

    filament_induced_velocity(
        filament_start_coords :: Matrix{<:Real},
        filament_end_coords :: Matrix{<:Real},
        filament_strengths :: Vector{<:Real},
        measurement_points :: Matrix{<:Real})

Compute the velocity induced in the flow field by vortex filaments. The third
multiple-multiple method may be GPU accelerated.

# Arguments
- `filament_start_coord` : The start coordinates of vortex filaments
- `filament_end_coord` : The end coordinates of vortex filaments
- `filament_strengths` : The vorticity per unit length of vortex filaments.
- `measurement_points` : The points where induced velocity is measured.

Vector arguments are expected to have length 3. Filament matrix arguments are
expected to have size N by 3. Measurement matrix arguments are expected to
have size M by 3. Returns an M by 3 matrix representing velocities.
"""
function filament_induced_velocity(
    filament_start_coord :: Vector{<:Real},
    filament_end_coord :: Vector{<:Real},
    filament_strength :: Real,
    measurement_point :: Vector{<:Real})

    check_filament_definition(
        filament_start_coord, filament_end_coord, filament_strength)
    convertable_to_Vec3f_vect(measurement_point, "measurement_point")

    inducing_filament = VortexFilament(filament_start_coord, filament_end_coord, 
        filament_strength)
    mes_pnt = Vec3f(measurement_point)

    ret = Vec3f(0., 0., 0.)
    #=
    CVTX_EXPORT bsv_V3f cvtx_F3D_S2S_vel(
        const cvtx_F3D *self,
        const bsv_V3f mes_point);
    =#
    ret = ccall(
            ("cvtx_F3D_S2S_vel", libcvortex), 
            Vec3f, 
            (Ref{VortexFilament}, Vec3f),
            inducing_filament, mes_pnt)
    return Vector{Float32}(ret)
end

function filament_induced_velocity(
    filament_start_coords :: Matrix{<:Real},
    filament_end_coords :: Matrix{<:Real},
    filament_strengths :: Vector{<:Real},
    measurement_point :: Vector{<:Real})

    check_filament_definition(
        filament_start_coords, filament_end_coords, filament_strengths)
    convertable_to_Vec3f_vect(measurement_point, "measurement_point")

    ni = size(filament_end_coords)[1]
    inducing_filaments = map(
        i->VortexFilament(filament_start_coords[i,:], filament_end_coords[i, :], 
        filament_strengths[i]),
        1:ni)
    mes_pnt = Vec3f(measurement_point)

    pargarr = Vector{Ptr{VortexFilament}}(undef, ni)
    for i = 1 : ni
        pargarr[i] = Base.pointer(inducing_filaments, i)
    end
    ret = Vec3f(0., 0., 0.)
    #=
    CVTX_EXPORT bsv_V3f cvtx_F3D_M2S_vel(
        const cvtx_F3D **array_start,
        const int num_filaments,
    =#		
    ret = ccall(
            ("cvtx_F3D_M2S_vel", libcvortex), 
            Vec3f, 
            (Ref{Ptr{VortexFilament}}, Cint, Vec3f),
            pargarr, ni, mes_pnt
            )
    return Vector{Float32}(ret)
end

function filament_induced_velocity(
    filament_start_coords :: Matrix{<:Real},
    filament_end_coords :: Matrix{<:Real},
    filament_strengths :: Vector{<:Real},
    measurement_points :: Matrix{<:Real})

    check_filament_definition(
        filament_start_coords, filament_end_coords, filament_strengths)
    convertable_to_Vec3f_vect(measurement_points, "measurement_points")

    ni = size(filament_end_coords)[1]
    np = size(measurement_points)[1]
    inducing_filaments = map(
        i->VortexFilament(filament_start_coords[i,:], filament_end_coords[i, :], 
            filament_strengths[i]),
        1:ni)
    mes_pnts = map(i->Vec3f(measurement_points[i, :]), 1:np)

    pargarr = Vector{Ptr{VortexFilament}}(undef, ni)
    for i = 1 : length(pargarr)
        pargarr[i] = Base.pointer(inducing_filaments, i)
    end
    ret = Vector{Vec3f}(undef, np)
    #=
    CVTX_EXPORT void cvtx_F3D_M2M_vel(
        const cvtx_F3D **array_start,
        const int num_filaments,
        const bsv_V3f *mes_start,
        const int num_mes,
        bsv_V3f *result_array);
    =#	
    ccall(
        ("cvtx_F3D_M2M_vel", libcvortex), 
        Cvoid, 
        (Ptr{Ptr{VortexFilament}}, Cint, Ptr{Vec3f}, 
            Cint, Ref{Vec3f}),
        pargarr, ni, mes_pnts, np, ret)
    return Matrix{Float32}(ret)
end

"""
    filament_induced_dvort(
        filament_start_coord :: Vector{<:Real},
        filament_end_coord :: Vector{<:Real},
        filament_strength :: Real,
        induced_particle_position :: Vector{<:Real},
        induced_particle_vorticity :: Vector{<:Real})

    filament_induced_dvort(
        filament_start_coords :: Matrix{<:Real},
        filament_end_coords :: Matrix{<:Real},
        filament_strengths :: Vector{<:Real},
        induced_particle_position :: Vector{<:Real},
        induced_particle_vorticity :: Vector{<:Real})

    filament_induced_dvort(
        filament_start_coords :: Matrix{<:Real},
        filament_end_coords :: Matrix{<:Real},
        filament_strengths :: Vector{<:Real},
        induced_particle_position :: Matrix{<:Real},
        induced_particle_vorticity :: Matrix{<:Real})

Compute the rate of change of vorticity of vortex particles induced by 
vortex filaments. Multiple-multiple method may be GPU accelerated. Modelled
as singular vortex particles and filaments.

# Arguments
- `filament_start_coord` : The start coordinates of vortex filaments
- `filament_end_coord` : The end coordinates of vortex filaments
- `filament_strengths` : The vorticity per unit length of vortex filaments.
- `induced_particle_position` : The positions of vortex particles.
- `induced_particle_vorticity` : The vorticities of vortex particles

Vector arguments are expected to have length 3. Filament matrix arguments are
expected to have size N by 3. Vortex particles matrix arguments are expected to
have size M by 3. Returns an M by 3 matrix representing particle vorticity 
derivatives.
"""
function filament_induced_dvort(
    filament_start_coord :: Vector{<:Real},
    filament_end_coord :: Vector{<:Real},
    filament_strength :: Real,
    induced_particle_position :: Vector{<:Real},
    induced_particle_vorticity :: Vector{<:Real})

    check_filament_definition(
        filament_start_coord, filament_end_coord, filament_strength)

    inducing_filament = VortexFilament(filament_start_coord, filament_end_coord, 
        filament_strength)
    induced_particle = VortexParticle3D(
        induced_particle_position, induced_particle_vorticity, 0.0)

    ret = Vec3f(0., 0., 0.)
    #=
    CVTX_EXPORT bsv_V3f cvtx_F3D_S2S_dvort(
        const cvtx_F3D *self,
        const cvtx_P3D *induced_particle);
    =#
    ret = ccall(
            ("cvtx_F3D_S2S_dvort", libcvortex), 
            Vec3f, 
            (Ref{VortexFilament}, Ref{VortexParticle3D}),
            inducing_filament, induced_particle
            )
    return Vector{Float32}(ret)
end

function filament_induced_dvort(
    filament_start_coords :: Matrix{<:Real},
    filament_end_coords :: Matrix{<:Real},
    filament_strengths :: Vector{<:Real},
    induced_particle_position :: Vector{<:Real},
    induced_particle_vorticity :: Vector{<:Real})

    check_filament_definition(
        filament_start_coords, filament_end_coords, filament_strengths)

    ni = size(filament_end_coords)[1]
    inducing_filaments = map(
        i->VortexFilament(filament_start_coords[i,:], filament_end_coords[i, :], 
        filament_strengths[i]),
        1:ni)
    induced_particle = VortexParticle3D(
        induced_particle_position, induced_particle_vorticity, 0.0)

    pargarr = Vector{Ptr{VortexFilament}}(undef, length(inducing_filaments))
    for i = 1 : length(pargarr)
        pargarr[i] = Base.pointer(inducing_filaments, i)
    end
    #=
    CVTX_EXPORT bsv_V3f cvtx_F3D_M2S_dvort(
        const cvtx_F3D **array_start,
        const int num_filaments,
        const cvtx_P3D *induced_particle);
    =#
    ret = ccall(
            ("cvtx_F3D_M2S_dvort", libcvortex), 
            Vec3f, 
            (Ref{Ptr{VortexFilament}}, Cint, Ref{VortexParticle3D}),
            pargarr, length(inducing_filaments), induced_particle
            )
    return Vector{Float32}(ret)
end

function filament_induced_dvort(
    filament_start_coords :: Matrix{<:Real},
    filament_end_coords :: Matrix{<:Real},
    filament_strengths :: Vector{<:Real},
    induced_particle_position :: Matrix{<:Real},
    induced_particle_vorticity :: Matrix{<:Real})

    check_filament_definition(
        filament_start_coords, filament_end_coords, filament_strengths)

    ni = size(filament_end_coords)[1]
    np = size(induced_particle_position)[1]
    inducing_filaments = map(
        i->VortexFilament(filament_start_coords[i,:], filament_end_coords[i, :], 
        filament_strengths[i]),
        1:ni)
    induced_particles = map(
        i->VortexParticle3D(
            induced_particle_position[i, :], 
            induced_particle_vorticity[i, :], 0.0),
        1:np)

    pargarr = Vector{Ptr{VortexFilament}}(undef, ni)
    for i = 1 : length(pargarr)
        pargarr[i] = Base.pointer(inducing_filaments, i)
    end
    indarg = Vector{Ptr{VortexParticle3D}}(undef, np)
    for i = 1 : length(indarg)
        indarg[i] = Base.pointer(induced_particles, i)
    end
    ret = Vector{Vec3f}(undef, length(induced_particles))
    #=
    CVTX_EXPORT void cvtx_F3D_M2M_dvort(
        const cvtx_F3D **array_start,
        const int num_filaments,
        const cvtx_P3D **induced_start,
        const int num_induced,
        bsv_V3f *result_array);
    =#
    ccall(
        ("cvtx_F3D_M2M_dvort", libcvortex), 
        Cvoid, 
        (Ptr{Ptr{VortexFilament}}, Cint, Ptr{Ptr{VortexParticle3D}}, Cint, 
            Ptr{Vec3f}),
        pargarr, ni, indarg, np, ret)
    return Matrix{Float32}(ret)
end

"""
    induced_velocity_influence_matrix(
        filament_start_coords :: Matrix{<:Real},
        filament_end_coords :: Matrix{<:Real},
        measurement_points :: Matrix{<:Real},
        measurement_directions :: Matrix{<:Real})

The influence of vortex filaments on normal velocities at points in the
domain.

# Arguments
- `filament_start_coord` : The start coordinates of vortex filaments. Matrix 
of size N by 3.
- `filament_end_coord` : The end coordinates of vortex filaments. Matrix of 
size N by 3.
- `measurement_points` : The positions at which induced velocity is evaluated.
Matrix of size M by 3
- `measurement_directions` : The directions for which velocity compents are 
computed. Matrix of size M by 3

Returns a Matrix{Float32} with size (M, N).
"""
function filament_induced_velocity_influence_matrix(
    filament_start_coords :: Matrix{<:Real},
    filament_end_coords :: Matrix{<:Real},
    measurement_points :: Matrix{<:Real},
    measurement_directions :: Matrix{<:Real})

    check_filament_definition(
        filament_start_coords, filament_end_coords)
    @assert(size(measurement_points)[2]==3, "The size of the measurement "*
        "point vector must be N by 3. Actual size is ", 
        size(measurement_points), ".")
    @assert(size(measurement_directions)[2]==3, "The size of the measurement "*
        "direction vector must be N by 3. Actual size is ", 
        size(measurement_directions), ".")
    @assert(size(measurement_directions)==size(measurement_points),
        "The number of points defined by the measurement points and "*
        "measurement direction vectors should match, but instead measurement"*
        " directions define ", size(measurement_directions)[1], " directions "*
        "and measurement points define ", size(measurement_points), "points.")
    
    ni = size(filament_end_coords)[1]
    np = size(measurement_directions)[1]
    inducing_filaments = map(
        i->VortexFilament(filament_start_coords[i,:], filament_end_coords[i, :], 
        1.0),
        1:ni)
        
    mes_pnts = map(i->Vec3f(measurement_points[i, :]), 1:np)
    mes_dirs = map(i->Vec3f(measurement_directions[i, :]), 1:np)

    pargarr = Vector{Ptr{VortexFilament}}(undef, length(inducing_filaments))
    for i = 1 : length(pargarr)
        pargarr[i] = Base.pointer(inducing_filaments, i)
    end
    # Julia is column major, C is row major. 
    ret = Matrix{Float32}(undef, length(inducing_filaments), 
        length(mes_pnts))
    #=
    CVTX_EXPORT void cvtx_F3D_inf_mtrx(
        const cvtx_F3D **array_start,
        const int num_filaments,
        const bsv_V3f *mes_start,
        const bsv_V3f *dir_start,
        const int num_mes,
        float *result_matrix);
    =#
    ccall(
        ("cvtx_F3D_inf_mtrx", libcvortex), 
        Cvoid, 
        (Ptr{Ptr{VortexFilament}}, Cint, Ptr{Vec3f}, Ptr{Vec3f},
            Cint, Ptr{Float32}),
        pargarr, length(inducing_filaments), mes_pnts, mes_dirs, np, ret)
    return transpose(ret) # Fix row major -> column major
end
