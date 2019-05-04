##############################################################################
#
# Accelerators.jl
#
# Part of CVortex.jl
# Control use of accelerators such as GPUs.
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
	The number of GPU or other accelerators found by the CVortex library.

It is possible that accelerators may be listed multiple times if they can
be used by more than one installed platform.
To know how many are in use see number_of_enabled_accelerators()
"""
function number_of_accelerators()
	res = ccall(("cvtx_num_accelerators", libcvortex),
		Cint, ())
	return res
end

"""
	The number of accelerators that CVortex has been directed to use.

If no accelerators are in use this is zero, and the CPU is used for 
all computation.

Find which accelerators are enabled using accelerator_enabled,
and enable and disable with accelerator_enable and 
accelerator_disable.
"""
function number_of_enabled_accelerators()
	# int cvtx_num_enabled_accelerators();
	res = ccall(("cvtx_num_enabled_accelerators", libcvortex), Cint, ())
	return res
end

"""
	The name of an accelerator.

Input is an integer in the range 1:number_of_accelerators(). Returns
the name of the accelerator as a string.
"""
function accelerator_name(accelerator_id :: Int)
	@assert(accelerator_id >= 1, "Minimum accelerator id is 1.")
	@assert(accelerator_id <= number_of_accelerators(),
		"accelerator_id is higher than the number of accelerators found.")
	# char* cvtx_accelerator_name(int accelerator_id);
	res = ccall(("cvtx_accelerator_name", libcvortex), 
		Cstring, (Cint,), accelerator_id-1)
	return unsafe_string(res)
end

"""
	States whether CVortex is in use.
"""
function accelerator_enabled(accelerator_id :: Int)
	@assert(accelerator_id >= 1, "Minimum accelerator id is 1.")
	@assert(accelerator_id <= number_of_accelerators(),
		"accelerator_id is higher than the number of accelerators found.")
	# int cvtx_accelerator_enabled(int accelerator_id);
	res = ccall(("cvtx_accelerator_enabled", libcvortex), 
		Cint, (Cint,), accelerator_id-1)
	return res
end

"""
	Allows CVortex to use an accelerator.
"""
function accelerator_enable(accelerator_id :: Int)
	@assert(accelerator_id >= 1, "Minimum accelerator id is 1.")
	@assert(accelerator_id <= number_of_accelerators(),
		"accelerator_id is higher than the number of accelerators found.")
	# void cvtx_accelerator_enable(int accelerator_id);
	ccall(("cvtx_accelerator_enable", libcvortex), 
		Cvoid, (Cint,), accelerator_id-1)
	return
end

"""
	Stops CVortex using an accelerator.
"""
function accelerator_disable(accelerator_id :: Int)
	@assert(accelerator_id >= 1, "Minimum accelerator id is 1.")
	@assert(accelerator_id <= number_of_accelerators(),
		"accelerator_id is higher than the number of accelerators found.")
	# void cvtx_accelerator_disable(int accelerator_id);
	ccall(("cvtx_accelerator_disable", libcvortex), 
		Cvoid, (Cint,), accelerator_id-1)
	return
end
