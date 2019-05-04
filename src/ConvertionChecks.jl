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

function convertable_to_F32(arg :: Real, argname :: String)
    @assert(hasmethod(Float32, Tuple{typeof(arg)}),
        "Cannot convert "*argname*" of type ",
        typeof(arg), " to a Float32.")
    return
end

function convertable_to_F32(arg :: Vector{<:Real}, argname :: String)
    @assert(hasmethod(Float32, Tuple{eltype(arg)}),
        "Cannot convert elements  of "*argname*" of type ",
        typeof(arg), " to a Float32.")
    return
end

function convertable_to_F32(arg :: Matrix{<:Real}, argname :: String)
    @assert(hasmethod(Float32, Tuple{eltype(arg)}),
        "Cannot convert elements  of "*argname*" of type ",
        typeof(arg), " to a Float32.")
    return
end

function convertable_to_Vec3f_vect(arg :: Matrix{<:Real}, argname :: String)
    @assert(size(arg)[2]==3, "The argument "*argname*" should have size "*
        "N by 3, but actually has size ", size(arg), ".")
    convertable_to_F32(arg, argname)
end

function convertable_to_Vec3f_vect(arg :: Vector{<:Real}, argname :: String)
    @assert(length(arg)==3, "The argument "*argname*" should have size "*
        "3, but actually has size ", length(arg), ".")
    convertable_to_F32(arg, argname)
end

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
        "Filament start coordinate matrix must be N by 3. Actual size is ",
        size(filament_start_coord), ".")
    @assert(size(filament_end_coord)[2]==3, 
        "Filament end coordinate matrix must be N by 3. Actual size is ",
        size(filament_end_coord), ".")
    @assert(size(filament_start_coord)[1]==size(filament_end_coord)[1],
        "Both filament start coordinate and filament end coordinate "*
        "matrices must be N by 3 (ie. of equal size), but the sizes "*
        "don't match. filament_start_coord matrix is ",
        size(filament_start_coord), " and the end coordiante matrix is of"*
        "size ", size(filament_end_coord), ".")
    @assert(size(filament_start_coord)[1]==length(filament_strength),
        "The length of the filament strength vector does not match that "*
        "of the coordinate matrices. The filament coordinate matrices define"*
        " geometries for ", size(filament_start_coord)[1], " filaments ",
        "whilst the strength vector has length ", length(filament_strength),
        ".")
    convertable_to_F32(filament_strength, "filament_strength")
    convertable_to_F32(filament_start_coord, "filament_start_coord")
    convertable_to_F32(filament_end_coord, "filament_end_coord")
    return
end
