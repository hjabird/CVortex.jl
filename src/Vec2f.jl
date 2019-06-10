##############################################################################
#
# Vec2f.jl
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

"""
A 2D Float32 vector.
"""
struct Vec2f
    x :: Float32
    y :: Float32
end

function Vec2f(x::Real, y::Real)
    return Vec2f(Float32(x), Float32(y))
end

function Vec2f(a::Vector{<:Real})
    @assert(length(a)==2)
    return Vec2f(a[1], a[2])
end

function Vector{Vec2f}(a::Matrix{<:Real})
	@assert(size(a)[2] == 2, "Input matrix is expected to be n rows by 2 "*
		"columns. Actual input matrix size was " * string(size(a)))
	len = size(a)[1]
	v = Vector{Vec2f}(undef, len)
	for i = 1 : len
		v[i] = convert(Vec2f, a[i, :])
	end
	return v
end

function Vector{T}(a::Vec2f) where T<:Real
	return Vector{T}([a.x, a.y])
end

function Matrix{T}(a::Vector{Vec2f}) where T <: Real
	len = length(a)
	mat = Matrix{T}(undef, len, 2)
	for i = 1 : len
		mat[i, :] = [a[i].x, a[i].y]
	end
	return mat
end
