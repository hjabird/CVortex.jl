##############################################################################
#
# test_3d_vortex_particles.jl
#
# Check 3D vortex particles behave as expected.
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

function two_particles_basic_case_3d()
	@testset "3D 2 particle interaction" begin
		pl0 = [0,0,0]
		plz = [0,0,1]
		ply = [0,1,0]
		plx = [1,0,0]
		pvx = [1,0,0]
		pvy = [0,1,0]
		pvz = [0,0,1]

		# Singular kernel ---------------------------------------------------------------------------------------
		kernel = singular_regularisation()
		@test isapprox([0,-1,0]./(4*pi), particle_induced_velocity(pl0, pvx, plz, kernel, 1.); rtol=1e-5)
		@test isapprox([-1,0,0]./(4*pi), particle_induced_velocity(plz, pvy, pl0, kernel, 1.); rtol=1e-5)
		@test isapprox([0,0,0]./(4*pi), particle_induced_velocity(pl0, pvx, pl0, kernel, 1.); rtol=1e-5)
		@test isapprox([0,0,0]./(4*pi), particle_induced_velocity(plz, pvy, plz, kernel, 1.); rtol=1e-5)
		@test isapprox([0,0,0]./(4*pi), particle_induced_velocity(pl0, pvz, plz, kernel, 1.); rtol=1e-5)
		@test isapprox([0,0,0]./(4*pi), particle_induced_velocity(plz, pvz, pl0, kernel, 1.); rtol=1e-5)
		@test isapprox([0,-0.25,0]./(4*pi), particle_induced_velocity(pl0, pvx, 2*plz, kernel, 1.); rtol=1e-5)
		@test isapprox([-0.25,0,0]./(4*pi), particle_induced_velocity(2*plz, pvy, pl0, kernel, 1.); rtol=1e-5)
		@test isapprox([0,-2,0]./(4*pi), particle_induced_velocity(pl0, 2*pvx, plz, kernel, 1.); rtol=1e-5)
		@test isapprox([-2,0,0]./(4*pi), particle_induced_velocity(plz, 2*pvy, pl0, kernel, 1.); rtol=1e-5)
		@test isapprox([0,0,1]./(4*pi), particle_induced_velocity(pl0, pvx, ply, kernel, 1.); rtol=1e-5)
		@test isapprox([0,0,-1]./(4*pi), particle_induced_velocity(ply, pvx, pl0, kernel, 1.); rtol=1e-5)
		# Check these don't fail.
		kernel = gaussian_regularisation()
		kernel = planetary_regularisation()
		kernel = winckelmans_regularisation()
		# Check visc dvort.
		kernel = gaussian_regularisation()
		gaussr1 = sqrt(2/pi) * exp(-1/2)
		gaussr2 = sqrt(2/pi) * exp(-4/2)
		@test isapprox((pvz-pvy)*2*gaussr1, 
			particle_visc_induced_dvort(ply, pvz, 1, pl0, pvy, 1, kernel, 1., 1.); rtol=1e-5)
		@test isapprox((pvx-pvy)*2*gaussr1, 
			particle_visc_induced_dvort(plx, pvx, 1, pl0, pvy, 1, kernel, 1., 1.); rtol=1e-5)
		@test isapprox((pvz-pvz)*2*gaussr1, 
			particle_visc_induced_dvort(ply, pvz, 1, pl0, pvz, 1, kernel, 1., 1.); rtol=1e-5)
		@test isapprox((pvz-pvy)*2*gaussr2, 
			particle_visc_induced_dvort(2*ply, pvz, 1, pl0, pvy, 1, kernel, 1., 1.); rtol=1e-5)
		@test isapprox((pvx-pvy)*2*gaussr2, 
			particle_visc_induced_dvort(2*plx, pvx, 1, pl0, pvy, 1, kernel, 1., 1.); rtol=1e-5)
		@test isapprox((pvz-pvz)*2*gaussr2, 
			particle_visc_induced_dvort(2*ply, pvz, 1, pl0, pvz, 1, kernel, 1., 1.); rtol=1e-5)
		@test isapprox((pvz-pvy)*gaussr1, 
			particle_visc_induced_dvort(ply, pvz, 0.5, pl0, pvy, 0.5, kernel, 1., 1.); rtol=1e-5)
		@test isapprox((pvz-pvy)*gaussr1, 
			particle_visc_induced_dvort(ply, 0.5*pvz, 0.5, pl0, pvy, 1.0, kernel, 1., 1.); rtol=1e-5)
		@test isapprox((pvz-pvy)*4*gaussr1, 
			particle_visc_induced_dvort(ply, pvz, 1, pl0, pvy, 1, kernel, 1., 2.); rtol=1e-5)
		@test isapprox((pvz-pvy)*8*gaussr2, 
			particle_visc_induced_dvort(ply, pvz, 1, pl0, pvy, 1, kernel, 0.5, 1.); rtol=1e-5)
	end
end

function test_correct_output_matrix_dims_3d_vortex_particles()
	@testset "3D particle correct output dimensions" begin
		n1 = 31
		n2 = 47	
		kernel = singular_regularisation()
		@test (3,) == size(particle_induced_velocity(rand(3), rand(3), rand(3), kernel, 1.))
		@test (3,) == size(particle_induced_velocity(rand(n1, 3), rand(n1, 3), rand(3), kernel, 1.))
		@test (n1, 3) == size(particle_induced_velocity(rand(n1, 3), rand(n1, 3), rand(n1, 3), kernel, 1.))
		@test (n2, 3) == size(particle_induced_velocity(rand(n1, 3), rand(n1, 3), rand(n2, 3), kernel, 1.))
		@test (n1, 3) == size(particle_induced_velocity(rand(3), rand(3), rand(n1, 3), kernel, 1.))
		@test (3,) == size(particle_induced_dvort(rand(3), rand(3), rand(3), rand(3), kernel, 1.))
		@test (3,) == size(particle_induced_dvort(rand(n1, 3), rand(n1, 3), rand(3), rand(3), kernel, 1.))
		@test (n1, 3) == size(particle_induced_dvort(rand(n1, 3), rand(n1, 3), rand(n1, 3), rand(n1, 3), kernel, 1.))
		@test (n2, 3) == size(particle_induced_dvort(rand(n1, 3), rand(n1, 3), rand(n2, 3), rand(n2, 3), kernel, 1.))
		@test (n1, 3) == size(particle_induced_dvort(rand(3), rand(3), rand(n1, 3), rand(n1, 3), kernel, 1.))
		kernel = gaussian_regularisation()
		@test (3,) == size(particle_visc_induced_dvort(rand(3), rand(3), rand(), rand(3), rand(3), rand(), kernel, 1., 1))
		@test (3,) == size(particle_visc_induced_dvort(rand(n1, 3), rand(n1, 3), rand(n1), rand(3), rand(3), rand(), kernel, 1., 1))
		@test (n1, 3) == size(particle_visc_induced_dvort(rand(n1, 3), rand(n1, 3), rand(n1), rand(n1, 3), rand(n1, 3), rand(n1), kernel, 1., 1))
		@test (n2, 3) == size(particle_visc_induced_dvort(rand(n1, 3), rand(n1, 3), rand(n1), rand(n2, 3), rand(n2, 3), rand(n2), kernel, 1., 1))
		@test (n1, 3) == size(particle_visc_induced_dvort(rand(3), rand(3), rand(), rand(n1, 3), rand(n1, 3), rand(n1), kernel, 1., 1))
	end
end

test_correct_output_matrix_dims_3d_vortex_particles()
two_particles_basic_case_3d()
