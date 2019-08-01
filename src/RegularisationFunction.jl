##############################################################################
#
# RegularisationFunction.jl
#
# Part of cvortex.jl
# Get vortex regularisation methods.
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
A 3D vortex regularisation function.

The method by which the singular nature of a vortex particle is managed.
Generally, this structure is best obtained via *_regularisation(). For example
singular_regularisation(), planetary_regularisation(), 
gaussian_regularisation() and winckelmans_regulariation().

Exposure of this allows the use of user kernels in the CPU multithreaded
implementations in cvortex. If you're interested in this, looking at the
cvortex library and the source of cvortex.jl is suggested to understand
the required functions signitures. 

# Examples
```
using CVortex
my_regularisation_kernel = singular_regularisation()
my_regularisation_kernel = winckelmans_regularisation()
my_regularisation_kernel = planetary_regularisation()
my_regularisation_kernel = gaussian_regularisation()
```
"""
struct RegularisationFunction
	g_3D :: Ptr{Cvoid}			# Actually float(*g_3D)(float rho)
	g_2D :: Ptr{Cvoid}			# Actually float(*g_2D)(float rho)
	zeta_3D :: Ptr{Cvoid}		# Actually float(*zeta_3D)(float rho)
	combined_3D :: Ptr{Cvoid}	# Actually void(*combined_3D)(float rho, float* g, float* zeta)
	eta_3D :: Ptr{Cvoid}		# Actually float(*eta_3D)(float rho)
	eta_2D :: Ptr{Cvoid}		# Actually float(*eta_2D)(float rho)
	cl_kernel_name_ext :: NTuple{32, Cchar}	# Char[32]
end

#= Functions to to get VortexFunc structures =#
"""
Implements singular vortex particles.
returns RegularisationFunction struct.
"""
function singular_regularisation()
	ret = ccall(("cvtx_VortFunc_singular", libcvortex), 
		RegularisationFunction, ())
	return ret;
end
"""
Implements Winckelmans' high order algebraic regularisation.
returns RegularisationFunction struct.
"""
function winckelmans_regularisation()
	ret = ccall(("cvtx_VortFunc_winckelmans", libcvortex), 
		RegularisationFunction, ())
	return ret;
end
"""
Implements Planetary regularisation.
returns RegularisationFunction struct.
"""
function planetary_regularisation()
	ret = ccall(("cvtx_VortFunc_planetary", libcvortex), 
		RegularisationFunction, ())
	return ret;
end
"""
Implements Gaussian regularisation.
returns RegularisationFunction struct.
"""
function gaussian_regularisation()
	ret = ccall(("cvtx_VortFunc_gaussian", libcvortex), 
		RegularisationFunction, ())
	return ret;
end
