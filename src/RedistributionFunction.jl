##############################################################################
#
# RedistributionFunction.jl
#
# Part of cvortex.jl
# Get vortex redistribution methods.
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
A vortex redistribution function.

The method by which vortex particles are redistibuted in lagrangian vortex
particles onto a grid to ensure long term stability.

# Examples
```
using CVortex
my_redistribution_kernel = singular_regularisation()
my_redistribution_kernel = winckelmans_regularisation()
my_redistribution_kernel = planetary_regularisation()
my_redistribution_kernel = gaussian_regularisation()
```
"""
struct RedistributionFunction
	func:: Ptr{Cvoid}			# Actually float(*g_3D)(float rho)
	critical_dist :: Float32	# Char[32]
end

#= Functions to to get VortexFunc structures =#

"""
Implements 0th order redistribution method.
The use of this redistribution is not advised.
lambda3_redistribution or m4p_redistribution are advised.
returns RedistributionFunction struct.
"""
function lambda0_redistribution()
	ret = ccall(("cvtx_RedistFunc_lambda0", libcvortex), 
        RedistributionFunction, ())
	return ret;
end
"""
Implements 1st order redistribution method.
This method is dissipative.
returns RedistributionFunction struct.
"""
function lambda1_redistribution()
	ret = ccall(("cvtx_RedistFunc_lambda1", libcvortex), 
        RedistributionFunction, ())
	return ret;
end
"""
Implements 2nd order redistribution method.
This redistribution is discontinuous, and consequently numerically
less useful than the lambda3_redistribution or m4p_redistribution.
returns RedistributionFunction struct.
"""
function lambda2_redistribution()
	ret = ccall(("cvtx_RedistFunc_lambda2", libcvortex), 
        RedistributionFunction, ())
	return ret;
end
"""
Implements 3th order redistribution method.
returns RedistributionFunction struct.
"""
function lambda3_redistribution()
	ret = ccall(("cvtx_RedistFunc_lambda3", libcvortex), 
        RedistributionFunction, ())
	return ret;
end
"""
Implements M_4' method of vorticity redistribution.
returns RedistributionFunction struct.
"""
function m4p_redistribution()
	ret = ccall(("cvtx_RedistFunc_m4p", libcvortex), 
        RedistributionFunction, ())
	return ret;
end
