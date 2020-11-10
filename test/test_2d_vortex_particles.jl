##############################################################################
#
# test_2d_vortex_particles.jl
#
# Check 2D vortex particles behave as expected.
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

function two_particles_basic_case_2d()
	@testset "2D 2 particle interaction" begin
		pl0 = [0,0]
		ply = [0,1]
		plx = [1,0]

		# Singular kernel ---------------------------------------------------------------------------------------
		kernel = singular_regularisation()
		@test isapprox([1,0]./(2*pi), particle_induced_velocity(pl0, 1, ply, kernel, 1.); rtol=1e-5)
		@test isapprox([0,-1]./(2*pi), particle_induced_velocity(pl0, 1, plx, kernel, 1.); rtol=1e-5)
		@test isapprox([2,0]./(2*pi), particle_induced_velocity(pl0, 2, ply, kernel, 1.); rtol=1e-5)
        @test isapprox([0.5,0]./(2*pi), particle_induced_velocity(pl0, 1, 2*ply, kernel, 1.); rtol=1e-5)
	end
end


function test_correct_output_matrix_dims_2d_vortex_particles()
    @testset "2D particle correct output dimensions" begin
        n1 = 31
        n2 = 47	
        kernel = singular_regularisation()
        @test (2,) == size(particle_induced_velocity(rand(2), rand(), rand(2), kernel, 1.))
        @test (2,) == size(particle_induced_velocity(rand(n1, 2), rand(n1), rand(2), kernel, 1.))
        @test (n1, 2) == size(particle_induced_velocity(rand(n1, 2), rand(n1), rand(n1, 2), kernel, 1.))
        @test (n2, 2) == size(particle_induced_velocity(rand(n1, 2), rand(n1), rand(n2, 2), kernel, 1.))
        @test (n1, 2) == size(particle_induced_velocity(rand(2), rand(), rand(n1, 2), kernel, 1.))
        kernel = gaussian_regularisation()
        @test Float32 == typeof(particle_visc_induced_dvort(
            rand(2), rand(), rand(), rand(2), rand(), rand(), kernel, 1., 1))
        @test Float32 == typeof(particle_visc_induced_dvort(
            rand(n1, 2), rand(n1), rand(n1), rand(2), rand(), rand(), kernel, 1., 1))
        @test (n1,) == size(particle_visc_induced_dvort(
            rand(n1, 2), rand(n1), rand(n1), rand(n1, 2), rand(n1), rand(n1), kernel, 1., 1))
        @test (n2,) == size(particle_visc_induced_dvort(
            rand(n1, 2), rand(n1), rand(n1), rand(n2, 2), rand(n2), rand(n2), kernel, 1., 1))
        @test (n1,) == size(particle_visc_induced_dvort(
            rand(2), rand(), rand(), rand(n1, 2), rand(n1), rand(n1), kernel, 1., 1))
    end
end

test_correct_output_matrix_dims_2d_vortex_particles()
two_particles_basic_case_2d()
