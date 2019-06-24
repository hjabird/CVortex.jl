##############################################################################
#
# ConvertionChecks.jl
#
# Part of CVortex.jl
# Checks that a type can be successfully converted to the types required
# for the underlying cvortex library, giving useful error messages.
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
    Gives an assert message if something cannot be converted to something
    of type or eltype Float32.

Part of CVortex. Not for client use.
"""
function convertable_to_F32(arg :: Real, argname :: String)
    @assert(hasmethod(Float32, Tuple{typeof(arg)}),
        "Cannot convert "*argname*" of type "*
        string(typeof(arg))*" to a Float32.")
    return
end

function convertable_to_F32(arg :: Vector{<:Real}, argname :: String)
    @assert(hasmethod(Float32, Tuple{eltype(arg)}),
        "Cannot convert elements  of "*argname*" of type "*
        string(typeof(arg))*" to a Float32.")
    return
end

function convertable_to_F32(arg :: Matrix{<:Real}, argname :: String)
    @assert(hasmethod(Float32, Tuple{eltype(arg)}),
        "Cannot convert elements  of "*argname*" of type "*
		string(typeof(arg))*" to a Float32.")
    return
end

"""
    Gives an assert message if something cannot be converted to a 
    CVortex.Vec3f or Vector{CVortex.Vec3f}.

Part of CVortex. Not for client use.
"""
function convertable_to_Vec3f_vect(arg :: Matrix{<:Real}, argname :: String)
    @assert(size(arg)[2]==3, "The argument "*argname*" should have size "*
        "N by 3, but actually has size "*string(size(arg))*".")
    convertable_to_F32(arg, argname)
    return
end

function convertable_to_Vec3f_vect(arg :: Vector{<:Real}, argname :: String)
    @assert(length(arg)==3, "The argument "*argname*" should have size "*
        "3, but actually has size "*string(length(arg))*".")
    convertable_to_F32(arg, argname)
    return
end

"""
    Gives an assert message if something cannot be converted to a 
    CVortex.Vec2f or Vector{CVortex.Vec2f}.

Part of CVortex. Not for client use.
"""
function convertable_to_Vec2f_vect(arg :: Matrix{<:Real}, argname :: String)
    @assert(size(arg)[2]==2, "The argument "*argname*" should have size "*
        "N by 2, but actually has size "*string(size(arg))*".")
    convertable_to_F32(arg, argname)
    return
end

function convertable_to_Vec2f_vect(arg :: Vector{<:Real}, argname :: String)
    @assert(length(arg)==2, "The argument "*argname*" should have size "*
        "2, but actually has size "*string(length(arg))*".")
    convertable_to_F32(arg, argname)
    return
end

"""
    Gives an assert message if something cannot be converted to a 
    CVortex.VortexFilament or Vector{CVortex.VortexFilament}.

Part of CVortex. Not for client use.
"""
function check_filament_definition(
    filament_start_coord :: Vector{<:Real},
    filament_end_coord :: Vector{<:Real},
    filament_strength :: Real)

    @assert(length(filament_start_coord)==3, 
        "Filament start coordinate vector must have length 3.")
    @assert(length(filament_end_coord)==3, 
        "Filament end coordinate vector must have length 3.")
    convertable_to_F32(filament_strength, "filament_strength")
    convertable_to_F32(filament_start_coord, "filament_start_coord")
    convertable_to_F32(filament_end_coord, "filament_end_coord")
    return
end

function check_filament_definition(
    filament_start_coord :: Matrix{<:Real},
    filament_end_coord :: Matrix{<:Real},
    filament_strength :: Vector{<:Real})

    @assert(size(filament_start_coord)[2]==3, 
        "Filament start coordinate matrix must be N by 3. Actual size is "*
        string(size(filament_start_coord))*".")
    @assert(size(filament_end_coord)[2]==3, 
        "Filament end coordinate matrix must be N by 3. Actual size is "*
        string(size(filament_end_coord))*".")
    @assert(size(filament_start_coord)[1]==size(filament_end_coord)[1],
        "Both filament start coordinate and filament end coordinate "*
        "matrices must be N by 3 (ie. of equal size), but the sizes "*
        "don't match. filament_start_coord matrix is "*
        string(size(filament_start_coord))*" and the end coordiante matrix is of"*
        "size "*string(size(filament_end_coord))*".")
    @assert(size(filament_start_coord)[1]==length(filament_strength),
        "The length of the filament strength vector does not match that "*
        "of the coordinate matrices. The filament coordinate matrices define"*
        " geometries for "* string(size(filament_start_coord)[1]) * 
        " filaments whilst the strength vector has length " * 
        string(length(filament_strength)) * ".")
    convertable_to_F32(filament_strength, "filament_strength")
    convertable_to_F32(filament_start_coord, "filament_start_coord")
    convertable_to_F32(filament_end_coord, "filament_end_coord")
    return
end

function check_filament_definition(
    filament_start_coord :: Matrix{<:Real},
    filament_end_coord :: Matrix{<:Real})

    @assert(size(filament_start_coord)[2]==3, 
        "Filament start coordinate matrix must be N by 3. Actual size is "*
        string(size(filament_start_coord))*".")
    @assert(size(filament_end_coord)[2]==3, 
        "Filament end coordinate matrix must be N by 3. Actual size is "*
        string(size(filament_end_coord))*".")
    @assert(size(filament_start_coord)[1]==size(filament_end_coord)[1],
        "Both filament start coordinate and filament end coordinate "*
        "matrices must be N by 3 (ie. of equal size), but the sizes "*
        "don't match. filament_start_coord matrix is "*
        string(size(filament_start_coord))*" and the end coordiante matrix is of"*
        "size "*string(size(filament_end_coord))*".")
    convertable_to_F32(filament_start_coord, "filament_start_coord")
    convertable_to_F32(filament_end_coord, "filament_end_coord")
    return
end

"""
    Gives an assert message if something cannot be converted to a 
    CVortex.VortexParticle or Vector{CVortex.VortexParticle}.

Part of CVortex. Not for client use.
"""
function check_particle_definition_3D(
    particle_coords :: Matrix{<:Real},
    particle_vorts  :: Matrix{<:Real})

    convertable_to_Vec3f_vect(particle_coords, "particle_coords")
    convertable_to_Vec3f_vect(particle_vorts, "particle_vorts")
    @assert(size(particle_coords)==size(particle_vorts),
        "The number of particle defined by particle_coords does not "*
        "match that defined by particle_vorts. particle_coords defines "*
		string(size(particle_coords)[1])*" particle coordinates whilst "*
        "particle_vorts defines "*string(size(particle_vorts))* 
        " particle vorticities.")
    return
end

function check_particle_definition_3D(
    particle_coords :: Matrix{<:Real},
    particle_vorts  :: Matrix{<:Real},
    particle_vols :: Vector{<:Real})

    convertable_to_Vec3f_vect(particle_coords, "particle_coords")
    convertable_to_Vec3f_vect(particle_vorts, "particle_vorts")
    convertable_to_F32(particle_vols, "particle_vols")
    @assert(size(particle_coords)==size(particle_vorts),
        "The number of particle defined by particle_coords does not "*
        "match that defined by particle_vorts. particle_coords defines "*
		string(size(particle_coords)[1])*" particle coordinates whilst "*
        "particle_vorts defines "*string(size(particle_vorts))*
        " particle vorticities.")
    @assert(length(particle_vols)==size(particle_coords)[1],
        "The number of particles defined by the particle volumes vector"*
        " does not match that of the coordinate and vorticity matrices. "*
        "The volume vector is of length "*string(length(particle_vols))*
        " and particle_coords suggests "*string(size(particle_coords)[1])*
        " particles.")
    return
end

function check_particle_definition_3D(
    particle_coords :: Vector{<:Real},
    particle_vorts  :: Vector{<:Real})

    convertable_to_Vec3f_vect(particle_coords, "particle_coords")
    convertable_to_Vec3f_vect(particle_vorts, "particle_vorts")
    return
end

function check_particle_definition_3D(
    particle_coords :: Vector{<:Real},
    particle_vorts  :: Vector{<:Real},
    particle_vol :: Real)

    convertable_to_Vec3f_vect(particle_coords, "particle_coords")
    convertable_to_Vec3f_vect(particle_vorts, "particle_vorts")
    convertable_to_F32(particle_vol, "particle_vol")
    return
end

"""
    Gives an assert message if something cannot be converted to a 
    CVortex.VortexParticle2D or Vector{CVortex.VortexParticle2D}.

Part of CVortex. Not for client use.
"""
function check_particle_definition_2D(
    particle_coords :: Matrix{<:Real},
    particle_vorts  :: Vector{<:Real})

    convertable_to_Vec2f_vect(particle_coords, "particle_coords")
    convertable_to_F32(particle_vorts, "particle_vorts")
    @assert(size(particle_coords)[1]==length(particle_vorts),
        "The number of particle defined by particle_coords does not "*
        "match that defined by particle_vorts. particle_coords defines "*
		string(size(particle_coords)[1])*" particle coordinates whilst "*
        "particle_vorts defines "*string(length(particle_vorts))*
        " particle vorticities.")
    return
end

function check_particle_definition_2D(
    particle_coords :: Matrix{<:Real},
    particle_vorts  :: Vector{<:Real},
    particle_vols :: Vector{<:Real})

    convertable_to_Vec2f_vect(particle_coords, "particle_coords")
    convertable_to_F32(particle_vorts, "particle_vorts")
    convertable_to_F32(particle_vols, "particle_vols")
    @assert(size(particle_coords)[1]==length(particle_vorts),
        "The number of particle defined by particle_coords does not "*
        "match that defined by particle_vorts. particle_coords defines "*
		string(size(particle_coords)[1])* " particle coordinates whilst "*
        "particle_vorts defines "*string(length(particle_vorts))*
        " particle vorticities.")
    @assert(length(particle_vols)==size(particle_coords)[1],
        "The number of particles defined by the particle volumes vector"*
        " does not match that of the coordinate and vorticity matrices. "*
        "The volume vector is of length ", length(particle_vols),
        " and particle_coords suggests "*string(size(particle_coords)[1])*
        " particles.")
    return
end

function check_particle_definition_2D(
    particle_coords :: Vector{<:Real},
    particle_vort  :: Real)

    convertable_to_Vec2f_vect(particle_coords, "particle_coords")
    convertable_to_F32(particle_vort, "particle_vorts")
    return
end

function check_particle_definition_2D(
    particle_coords :: Vector{<:Real},
    particle_vort  :: Real,
    particle_vol :: Real)

    convertable_to_Vec2f_vect(particle_coords, "particle_coords")
    convertable_to_F32(particle_vort, "particle_vort")
    convertable_to_F32(particle_vol, "particle_vol")
    return
end
