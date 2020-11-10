##############################################################################
#
# test_accelerators.jl
#
# Test accelerator control functionality & binary info functionality.
#
# Copyright 2020 HJA Bird
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

let
	@testset "Accelerator information / control" begin
		info_string = CVortex.cvtx_binary_info()

		num_accl = number_of_accelerators()
		if occursin("using OpenCL: FALSE", info_string)
			@test num_accl == 0
		elseif occursin("using OpenCL: TRUE", info_string)
			@test num_accl >= 0
			@test num_accl < 10
		else
			@assert(false)
		end
		
		for i = 1 : num_accl
			@test accelerator_enabled(i) == 1
			accelerator_disable(i)
			@test accelerator_enabled(i) == 0
			accelerator_enable(i)
			@test accelerator_enabled(i) == 1	
			@test accelerator_name(i) != ""
		end
		
		@test_throws AssertionError accelerator_name(-2)
		@test_throws AssertionError accelerator_name(num_accl+1)
		@test_throws AssertionError accelerator_enable(-2)
		@test_throws AssertionError accelerator_enable(num_accl+1)
		@test_throws AssertionError accelerator_disable(-2)
		@test_throws AssertionError accelerator_disable(num_accl+1)
		@test_throws AssertionError accelerator_enabled(-2)
		@test_throws AssertionError accelerator_enabled(num_accl+1)
	end
end
