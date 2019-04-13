
push!(LOAD_PATH, "src/")
import WriteVTK
import cvortex

println("Running...")
let
    n_rings = 120
    n_per_ring = 200
    total_particles = n_rings * n_per_ring
    radius = 0.5
    length = 1
    thickness = 0.05
    str = pi * 0.1

    n_steps = 40
    dt = 0.05
    println("Total number of particles is ", total_particles, ".")
    kernel = cvortex.VortFunc_winckelmans()

    # Create a load of vortex particles in rings.
    particles = Vector{cvortex.VortexParticle}(undef, total_particles)
    regularisation_rad = 1.5 * pi * radius / n_per_ring
    particle_volume = 2*pi*radius * thickness * (length / n_rings) / n_per_ring
    acc = 1
    for ring_idx = 1 : n_rings
        for p_in_ring_idx = 1 : n_per_ring
            xloc = ring_idx * length / n_rings
            yloc = radius * cos(2 * pi * p_in_ring_idx / n_per_ring)
            zloc = radius * sin(2 * pi * p_in_ring_idx / n_per_ring)
            xvort = 0;
            yvort = str * -sin(2 * pi * p_in_ring_idx / n_per_ring) / (n_per_ring * n_rings)
            zvort = str * cos(2 * pi * p_in_ring_idx / n_per_ring) / (n_per_ring * n_rings)
            coord = cvortex.Vec3f(xloc, yloc, zloc)
            vort = cvortex.Vec3f(xvort, yvort, zvort)
            particles[acc] = cvortex.VortexParticle(
                coord, vort, particle_volume)
            acc += 1
        end
    end

    # A method to save the simulation to a file
    function save_particles(particles, step)
        points = zeros(3, total_particles)
        cells = Vector{WriteVTK.MeshCell}(undef, total_particles)
        for i = 1 : total_particles
            points[1, i] = particles[i].coord.x
            points[2, i] = particles[i].coord.y
            points[3, i] = particles[i].coord.z
            cells[i] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_VERTEX, [i])
        end
        vtkfile = WriteVTK.vtk_grid("particles"*string(step), points, cells)
        WriteVTK.vtk_save(vtkfile)
    end

    save_particles(particles, 0)
    for i = 1 : n_steps
        println("Step "*string(i)*".")
        vels = cvortex.induced_velocity(
            particles,
            [p.coord for p in particles],
            kernel,
            regularisation_rad
        )
        dvorts = cvortex.induced_dvort(
            particles,
            particles,
            kernel,
            regularisation_rad
        )

        for j = 1 : total_particles
            particles[j] = cvortex.VortexParticle(
                cvortex.Vec3f(
                    particles[j].coord.x + vels[j].x * dt,
                    particles[j].coord.y + vels[j].y * dt,
                    particles[j].coord.z + vels[j].z * dt),
                cvortex.Vec3f(
                    particles[j].vorticity.x + dvorts[j].x * dt,
                    particles[j].vorticity.y + dvorts[j].y * dt,
                    particles[j].vorticity.z + dvorts[j].z * dt),
                particle_volume)
        end
        if i % 1 == 0
            save_particles(particles, i)
        end
    end
end
