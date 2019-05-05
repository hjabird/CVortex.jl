##############################################################################
#
# particle_rings.jl
#
# A demo of using vortex particle ring interaction based on the work 
# Winckelmans and Leonard, J_Comp_Phys, 1993, Contributions to the 
# Vortex Particle Methods for the Computation of Three-Dimensional 
# Incompressible Unsteady Flows
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

    n_steps = 800
    dt = 0.01
    regdist = 0.2
    kernel = winckelmans_regularisation()

    function make_ring(section_radius::Real, radius::Real, vorticity::Real,
        circ_particles::Int, layers::Int)
        local theta_r = map(i->2 * pi * (i-.5) / circ_particles, 1:circ_particles)
        local particles = zeros(length(theta_r), 3)
        local vorts = zeros(length(theta_r), 3)
        local r_l = section_radius/layers
        local cell_area = pi * r_l^2
        local cross_section_area = pi * section_radius^2
        local cell_vpl = vorticity * cell_area * pi * radius^2 / 
                (cross_section_area * circ_particles)
        function sub_ring(theta_s, rc)
            local p=mapreduce(
                theta->[(radius+rc*cos(theta_s))*cos(theta), (radius+rc*cos(theta_s))*sin(theta), rc*sin(theta_s)]', 
                vcat, theta_r)
            local v=mapreduce(theta->cell_vpl * [-sin(theta), cos(theta), 0]', 
                vcat, theta_r)
            return p, v
        end

        # Ring centre.
        local particles, vorts = sub_ring(0., 0.)
        println("Made centre")
        # Outer layers
        for i = 1 : layers-1
            r_c = r_l * (1 + 12*i^2)/(6 * i)
            nr = 8 * i
            theta_s = map(j->2 * pi * j / nr, 1:nr)
            for theta in theta_s
                ps, vs = sub_ring(theta, r_c)
                particles=vcat(particles, ps)
                vorts=vcat(vorts, vs)
            end
        end
        return particles, vorts
    end
    function translate(points, dx)
        for i in 1 : size(points)[1]
            points[i, :] += dx
        end
        return points
    end
    function rotate(points, vorts, theta_x::Real, centre::Vector{<:Real})
        points = translate(points, -1 .*centre)
        mat = [1 0 0; 0 cos(theta_x) -sin(theta_x); 0 sin(theta_x) cos(theta_x)]
        for i in 1 : size(points)[1]
            points[i, :] = mat * points[i, :]
            vorts[i, :] = mat * vorts[i, :]
        end
        points = translate(points, centre)
        return points, vorts
    end

    particle_pos1, vorticities1 = make_ring(0.2, 1., -1., 64, 4)
    particle_pos2 = deepcopy(particle_pos1)
    vorticities2 = deepcopy(vorticities1)
    particle_pos1, vorticities1 = rotate(particle_pos1, vorticities1, deg2rad(15), [0., 0., 0.])
    particle_pos1 = translate(particle_pos1, [0., -1.35, 0.])
    particle_pos2, vorticities2 = rotate(particle_pos2, vorticities2, deg2rad(-15), [0., 0., 0.])
    particle_pos2 = translate(particle_pos2, [0., 1.35, 0.])

    particle_pos = vcat(particle_pos1, particle_pos2)
    particle_vorts = vcat(vorticities1, vorticities2)
    
    total_particles = size(particle_pos)[1]
    println("Total of ", total_particles, " particles.")

    # A method to save the simulation to a file
    function save_particles(step)
        cells = Vector{WriteVTK.MeshCell}(undef, total_particles)
        cells = map(
            x->WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_VERTEX, [x]), 
            1:size(particle_pos)[1])
        vtkfile = WriteVTK.vtk_grid("particles"*string(step), 
            particle_pos', cells)
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
        if i % 10 == 0
            save_particles(i)
        end
    end
    
end
