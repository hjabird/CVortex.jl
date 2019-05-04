##############################################################################
#
# Vec3f.jl
#
# Part of CVortex.jl
# A fixed size vector (needed for C equivalent structure.)
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

#import Base

"""
A 3D Float32 vector.
"""
struct Vec3f
    x :: Float32
    y :: Float32
    z :: Float32
end

function Vec3f(x::Real, y::Real, z::Real)
    return Vec3f(Float32(x), Float32(y), Float32(z))
end

function Vec3f(a::Vector{<:Real})
    @assert(length(a)==3)
    return Vec3f(a[1], a[2], a[3])
end

function Vector{Vec3f}(a::Matrix{<:Real})
	@assert(size(a)[2] == 3, "Input matrix is expected to be n rows by 3 "*
		"columns. Actual input matrix size was " * string(size(a)))
	len = size(a)[1]
	v = Vector{Vec3f}(undef, len)
	for i = 1 : len
		v[i] = convert(Vec3f, a[i, :])
	end
	return v
end

function Vector{T}(a::Vec3f) where T<:Real
	return Vector{T}([a.x, a.y, a.z])
end

function Matrix{T}(a::Vector{Vec3f}) where T <: Real
	len = length(a)
	mat = Matrix{T}(undef, len, 3)
	for i = 1 : len
		mat[i, :] = [a[i].x, a[i].y, a[i].z]
	end
	return mat
end
