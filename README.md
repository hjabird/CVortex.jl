# CVortex.jl

A Julia wrapper for GPU accelerated vortex filament and vortex particle methods of the CVortex library.

## Introduction

### What does CVortex.jl do?

CVortex.jl is a wrapper for the CVortex library. It has the following functionality:

* Compute velocities induced by collections of 2D regularised vortex particles, 3D regularised vortex particles and 3D straight singular vortex filaments.
* Compute the vortex stretching term induced on 3D vortex particles by other 3D particles or vortex filaments.
* Compute the change in particle vorticity due to viscous interaction between 3D regularised vortex particles.
* Redistribute 2D and 3D vortex particles onto a grid.

### What will it run on?

CVortex.jl only runs on 64bit Windows or 64bit Linux. The library is not compatible with MacOS or other 
CPU instruction sets.

To obtain the maximum benefit you'll also need an OpenCL 1.2 compatible GPU or iGPU. This includes:

* Intel integrated graphics 
* AMD integrated graphics and discrete GPUs
* Nvidia GPUs (Any GPU that runs CUDA)

You'll also have needed to have installed the appropriate GPU drivers. Note that
many hypervisors (programs that allow you to run virtual machines such as VirtualBox)
don't pass through graphics hardware. 

Even if you don't have compatable a compatible GPU, you'll still benefit 
from the multicore implementation.

### How can I get CVortex.jl?

You'll need to add the package to your system.
Run Julia and:
```
(v1.1) pkg> add https://github.com/hjabird/CVortex.jl
```
remembering that to access to package environment the `]` must be used.
Binaries for the CVortex library will automatically be downloaded.

### Is there documentation?

Yes! The ordinary help syntax within Julia (Type `?` within the REPL) will give you help on a particular topic. For example:
```
help?> particle_induced_velocity
```

## Using CVortex.jl

The first thing you'll need to do is to import CVortex.jl. This can be done using 
```
using CVortex
```

### Vortex filaments

All vortex filaments in CVortex are straight and singular. They have three properties, a start point, an
end point and a vorticity. The first two are 3D, and the latter is a scalar. 

#### Velocity
To obtain the velocity induced on a point in 3D one uses:
```
startp = rand(3)  # Filament start coordinate
endp = rand(3)    # Filament end coordinate
fvort = rand()    # A scalar. Filament's vorticity
mesp = rand(3)    # Velocity measurement location.
vel = filament_induced_velocity(startp, endp, fvort, mesp);
```
The returned `vel` is a Float32 array of length 3.

We use our hardware better if we group computations together. Suppose we had `N` vortex filaments, we can vectorise
the computation of the influence on a measurement point as
```
N = 10000
startps = rand(N, 3)
endps = rand(N, 3)
fvorts = rand(N)
mesp = rand(3)
vel = filament_induced_velocity(startps, endps, fvorts, mesp)
```
Again, `vel` is a Float32 array of length 3.

To create a problem suitable for GPU accelleration, we need multiple-multiple problems.
To do this the measurement points becomes a matrix:
```
N = 10000
M = 100000
startps = rand(N, 3)
endps = rand(N, 3)
fvorts = rand(N)
mesp = rand(M, 3)
vels = filament_induced_velocity(startps, endps, fvorts, mesp)
```
where `vels` is an M by 3 Float32 matrix.

#### Influence matrix

Its often desirable to obtain the influence of a vortex filament (perhaps it the context
of a vortex ring) on the velocity in a given direction at a point. A function for this is
included:
```
nvels = induced_velocity_influence_matrix(
      filament_start_coords :: Matrix{<:Real},
      filament_end_coords :: Matrix{<:Real},
      measurement_points :: Matrix{<:Real},
      measurement_directions :: Matrix{<:Real})
```
For N vortex filaments, a matrix A is returned such that, for a length N 
vector of filament vorticity called b, the induced velocities measured at 
the M points in M directions would be given by A*b.

### Vortex particles

Vortex particles are blobs of vorticity. Whilst they can be singular, this isn't good
for long term stability. CVortex therefore implements vortex particle regularisation.

#### Regularisation

CVortex.jl uses the struct `RegularisationFunction` to group together functions relevent 
to a regularisation method. A `RegularisationFunction` can be obtained using pre-programmed
routines:
```
singular_reg = singular_regularisation()
planet_reg = planetary_regularisation()
winckel_reg = winckelmans_regularisation()
gauss_reg = gaussian_regularisation()
```
`singular_regularisation()` isn't actually a regularisation because it allows the use of singular
vortex particles. `planetary_regularisation()` allow regularisation, but cannot be used
in viscous schemes. `winckelmans_regularisation()` is a higher order algebraic regularisation.
`gaussian_regularisation()` is normal gaussian regularisation. 

#### 2D vortex particles

Using 2D vortex particles is much like using singular vortex filament, with two additions:

* A regularisation function is required
* A regularisation distance is required

The regularisation distance is like the radius of the vortex particles. Roughly, it 
represents the finest fidelity that the field can resolve. Consequently, vortex particles
must overlap for a good solution.

##### Velocity
To obtain the velocity induced by a 2D vortex particle one uses
```
vels = particle_induced_velocity(particle_positions, particle_vorts, 
  measurement_points, regularisation, regularisation_distance)
```
Where:

* For single particle -> single measurement point: `particle_positions` is a length 2 vector, `particle_vorts` is scalar and `measurement_points` is a length 2 vector. `vels` is a length 2 vector.
* For multiple particles -> single measurement point: `particle_positions` is an N by 2 matrix, `particle_vorts` is a length N vector and `measurement_points` is a length 2 vector. `vels` is a length 2 vector.
* For multiple particles -> multiple measurement points: `particle_positions` is an N by 2 matrix, `particle_vorts` is a length N vector and `measurement_points` is an M by 2 matrix. `vels` is an M by 2 matrix.
In all cases, `regularisation_distance` is as scalar. Different sized particles aren't supported.

##### Viscous rate of change of vorticity
For viscous vortex particle method, the rate of change of vorticity of each particle is computed.
```
dvorts = particle_visc_induced_dvort(
  inducing_particle_positions, inducing_particle_vorts, inducing_particle_areas,
  induced_particle_positions, induced_particle_vorts, induced_particle_areas,
  regularisation, regularisation_distance, kinematic_viscosity)
```
Here the rate of change of vorticity on the "induced" particles is computed. The particle_area 
variables is of the same type as the vorticity vector.

#### 3D Vortex particles

3D vortex particles are characterised by a position (in 3D) and a vorticity vector (again, in 3D).
Like 2D particles, a regularisation function and a regularisation distance is required for
computation.

##### Velocity
To obtain the velocity induced by a 3D vortex particle one uses
```
vels = particle_induced_velocity(particle_positions, particle_vorts, 
  measurement_points, regularisation, regularisation_distance)
```
Where:

* For single particle -> single measurement point: `particle_positions` is a length 3 vector, `particle_vorts` is a length 3 vector and `measurement_points` is a length 3 vector. `vels` is a length 3 vector.
* For multiple particles -> single measurement point: `particle_positions` is an N by 3 matrix, `particle_vorts` is an N by 3 matrix and `measurement_points` is a length 3 vector. `vels` is a length 3 vector.
* For multiple particles -> multiple measurement points: `particle_positions` is an N by 3 matrix, `particle_vorts` is an N by 3 matrix and `measurement_points` is an M by 3 matrix. `vels` is an M by 3 matrix.
In all cases, `regularisation_distance` is as scalar. Different sized particles aren't supported.

#### Vortex stretching
In 3D vorticies can be "stretched" leading to a rate of change of vorticity. To compute this
the following is used:
```
dvort = particle_induced_dvort(
        inducing_particle_position, inducing_particle_vorticity,
        induced_particle_position, induced_particle_vorticity,
        kernel :: RegularisationFunction, regularisation_radius :: Real)
```

##### Viscous rate of change of vorticity
For viscous vortex particle method, the rate of change of vorticity of each particle is computed.
```
dvorts = particle_visc_induced_dvort(
  inducing_particle_positions, inducing_particle_vorts, inducing_particle_areas,
  induced_particle_positions, induced_particle_vorts, induced_particle_areas,
  regularisation, regularisation_distance, kinematic_viscosity)
```
Here the rate of change of vorticity on the "induced" particles is computed. The particle_area 
variables is of the same type as the vorticity vector.

#### Vortex particle redistribution

Vortex particles can be redistributed onto a grid to fix problems introduced by particles spreading out of 
grouping together.

To do this some kind of redistribution scheme is required. This scheme is encapsulated within
a `RedistributionFunction` struct. These can be created as
```
scheme = lambda0_redistribution()
scheme = lambda1_redistribution()
scheme = lambda2_redistribution()
scheme = lambda3_redistribution()
scheme = m4p_redistribution()
```
The `lambda3_redistribution()` scheme and  `m4p_redistribution()` scheme are generally recommended. 
The `lambda0_redistribution()` and `lambda2_redistribution()` are discontinious, and so can cause problems
numerically. The `lambda1_redistribution()` is dissipative.

Having chosen a scheme, particles can then be redistributed:
```
positions, vorts, areas = redistribute_particles_on_grid(
        particle_positions, inducing_particle_vorticity,
        redistribution_function, grid_density;
        negligible_vort=1e-4, max_new_particles=-1)
```
There are two optional parameters, `negligible_vort` and `max_new_particles`.
These are designed to stop lots of vortex particles with very small vorticities 
being created. 

`negligible_vort` is a threshold for discarding particles, and should be a 
value between 0 (discard no particles) and 1 (discard all particles). It is 
implemented as the proportion of the average particle's vorticity that any 
particle must have possess to be kept. The vorticity of any discarded particle
is distributed evenly among the remaining particles.

`max_new_particles` is a hard limit on the number of new vortex particles
that can be created. When equal to -1, there is no limit. If `negligible_vort` is
chosen such that there are more than the `max_new_particles` remaining, further 
particles are discarded until the number of particles is less than `max_new_particles`.
Due to the implementation, this may result in fewer particles than `max_new_particles`.

#### Mixing 3D vortex particles and filaments

It is possible to mix vortex particles and filaments in 3D. Since vortex filaments
are singular, it is not possible to include viscosity for these problems (unless 
you're willing to cheat somehow).

The only addition function required for putting both in one problem is the 
vortex stretching induced by vortex filaments on vortex particles. This can be
obtained using
```
dvort = filament_induced_dvort(
      filament_start_coord,
      filament_end_coord,
      filament_strength,
      induced_particle_position ,
      induced_particle_vorticity)
```
where everything is assumed to be singular.

### Contolling the accelerators / GPUs

In computers with multiple GPUs (probably an iGPU + discrete GPU) it may 
be desirable to control which GPU is being used, or in some cases to stop GPUs being used at
all.

But first, one must know how many GPUs CVortex has found:
```
number_of_gpus = number_of_accelerators()
```
where an integer is returned. 

The accelerators are given an index of 1:number_of_accelerators(). Acclerators
can then be controlled and investigated using the index. 

To obtain the name one uses:
```
name = accelerator_name(accelerator_index)
```
Note that the name may not be unique among your GPUs, or even share the name of the
product you purchased. For example, an for an AMD RX Vega 56:
```
julia> accelerator_name(1)
"gfx900"
```

To investigate whether CVortex is using a GPU:
```
in_use = accelerator_enabled(index)
```
which returns 1 (true) or 0 (false).

To enable an accelerator
```
accelerator_enable(index)
```
and to disable an accelerator
```
accelerator_disable(index)
```

## Potentially FAQ

***Why does CVortex.jl return Float32s?***
Because (almost) all the underlying code uses floats. GPU manufactures
cripple the double precision speed of their consumer-level GPUs, or may not include
double-precision capability at all (it isn't required by the OpenCL spec.). 
But since the discretisation of vortex particles of vortex filaments lead to more 
error than single precision computing, the cost of using single precision 
is negligible, whilst the compatability/performance benifits are huge.

***Why is the program hanging?***
The implementation of some OpenCL drivers is supposedly dodgy, especially
on older GPUs. 

***Why isn't CVortex.jl available on MacOS or X architecture?***
CVortex is theoretically compatable with MacOS and any CPU
architecture that can be targetet by MSVC, GCC or Clang. For
multithreading, OpenMP has to be available. For GPU acceleration,
an implementation of the full OpenCL 1.2 profile has to be available.
CVortex isn't compiled for these architectures or Mac, but I lack the hardware
on which to compile or test binaries. For a sufficiently keen bean,
it would be possible to compile cvortex from source, and copy the shared library
into CVortex.jl build directory.

***Why isn't it faster?***
It would be possible to make CVortex faster, but to do so would complicate
the interface. The code has also not be tailored to any particular hardware.

***Why does ```using CVortex``` take such a long time to run?***
```using CVortex``` calls the underlying CVortex library's initialisation
function which must in turn compile the OpenCL kernels for used by
the programs.

***Why isn't CVortex using my iGPU/GPU***
Thats potentially a big questions. If `number_of_accelerators()` returns the
expected number of devices, its probably because the problem isn't suitable for
GPU acceleration (only multiple-multiple problems are accelerated, and even 
then the problem must have a sufficient number of both target and source objects).
If the number is less than expected, things get more complicated. It may be that
the OpenCL kernels could not be sucessfully compiled (I've not encountered this)
or, more likely, the OpenCL ICD loader didn't find the device's OpenCL runtime library.
This might be because the drivers aren't properly installed. Additionally, on Windows,
driver installers are liable to overwrite files installed by other driver installers.
If you're running a virtual machine, check that the GPU is being passed through (if
the hypervisor is even capable of doing this).
