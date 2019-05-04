##############################################################################
#
# particle_vortex_tunnel.jl
#
# A demo of using vortex filament rings to create a tunnel.
#
##############################################################################

import WriteVTK
using CVortex
using Dates

let
    # We can print information about the system.
    println("Running particle_vortex_tunnel.jl")
    println("CVortex found ", number_of_accelerators(), " accelerators "
        *"on this system.")
    println("There are ", number_of_enabled_accelerators(), 
        " enabled accelerators. These are:")
    for i = 1 : number_of_enabled_accelerators()
        println(i, ":\t", accelerator_name(i))
    end

    # Set up the tubes's geometry:
    radius = 0.5
    len = 1
    # The discretisation
    n_rings = 40
    p_per_ring = 40
    # The strength of the vortex particles and the time step parameters
    str = 1 / n_rings
    n_steps = 40
    dt = 0.025
    # And the regularisation of the particle-particle interaction
    kernel = singular_regularisation()
    regdist = max(len / n_rings, 2 * pi * radius / p_per_ring) * 2

    total_points = n_rings * p_per_ring
    total_particles = n_rings * p_per_ring
    println("Total number of particles is ", total_particles, ".")

    # Create a load of vortex filaments in rings:
    particle_pos = zeros(total_particles, 3)
    particle_vorts = zeros(total_particles, 3)
    acc = 1
    for ring_idx = 1 : n_rings
        for p_in_ring_idx = 1 : p_per_ring
            xloc = ring_idx * len / n_rings
            yloc = radius * cos(2 * pi * p_in_ring_idx / p_per_ring)
            zloc = radius * sin(2 * pi * p_in_ring_idx / p_per_ring)
            yvort = str * 2 * pi * radius * -sin(2 * pi * p_in_ring_idx / p_per_ring) / p_per_ring
            zvort = str * 2 * pi * radius * cos(2 * pi * p_in_ring_idx / p_per_ring) / p_per_ring
            particle_pos[acc, :] = [xloc, yloc, zloc]
            particle_vorts[acc, :] = [0.0, yvort, zvort] 
            acc += 1
        end
    end

    # A method to save the simulation to a file
    function save_particles(step)
        cells = Vector{WriteVTK.MeshCell}(undef, total_particles)
        cells = map(
            x->WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_VERTEX, [x]), 
            1:size(particle_pos)[1])
        vtkfile = WriteVTK.vtk_grid("particles"*string(step), 
            transpose(particle_pos), cells)
        WriteVTK.vtk_save(vtkfile)
    end

    save_particles(0)
    print("Started.")
    ninter = total_particles^2
    for i = 1 : n_steps
        tstart = now()
        vels = particle_induced_velocity(particle_pos, particle_vorts, 
            particle_pos, kernel, regdist)
        dvorts = particle_induced_dvort(particle_pos, particle_vorts, 
            particle_pos, particle_vorts, kernel, regdist)
        tend = now()
        print("\rStep:\t", i, "\tInteractions per second: ", 
            1000*round(ninter / Float64((tend - tstart).value)),"\t\t\t")
        particle_pos .+= dt .* vels
        particle_vorts .+= dt * dvorts
        if i % 1 == 0
            save_particles(i)
        end
    end
end
