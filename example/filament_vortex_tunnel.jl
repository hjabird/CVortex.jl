##############################################################################
#
# filament_vortex_tunnel.jl
#
# A demo of using vortex filament rings to create a tunnel.
#
##############################################################################

import WriteVTK
using CVortex

let
    # We can print information about the system.
    println("Running filament_vortex_tunnel.jl")
    println("CVortex found ", number_of_accelerators(), " accelerators "
        *"on this system.")
    println("There are ", number_of_enabled_accelerators(), 
        " enabled accelerators. These are:")
    for i = 1 : number_of_enabled_accelerators()
        println(i, ":\t", accelerator_name(i))
    end

    # Set up the problem's geometry:
    radius = 0.5
    length = 1
    # The discretisation
    n_rings = 30
    p_per_ring = 30
    str = 0.05
    # And time stuff.
    n_steps = 40
    dt = 0.1
    
    f_per_ring = p_per_ring
    total_points = n_rings * p_per_ring
    total_fils = n_rings * f_per_ring
    println("Total number of straight filaments is ", total_fils, ".")

    # Create a load of vortex filaments in rings:
    points = zeros(total_points, 3)
    fil_starts = zeros(total_fils, 3)
    fil_ends = zeros(total_fils, 3)
    fil_strs = zeros(total_fils)
    acc = 1
    for ring_idx = 1 : n_rings
        for p_in_ring_idx = 1 : p_per_ring
            xloc = ring_idx * length / n_rings
            yloc = radius * cos(2 * pi * p_in_ring_idx / p_per_ring)
            zloc = radius * sin(2 * pi * p_in_ring_idx / p_per_ring)
            points[acc, :] = [xloc, yloc, zloc]
            acc += 1
        end
    end

    function points_to_fils()
        local acc = 1
        for ring_idx = 1 : n_rings
            for p_in_ring_idx = 1 : f_per_ring
                fil_starts[acc, :] = points[acc, :]
                if p_in_ring_idx == f_per_ring
                    fil_ends[acc, :] = points[acc - f_per_ring + 1, :]
                else
                    fil_ends[acc, :] = points[acc + 1, :]
                end
                fil_strs[acc] = str
                acc += 1
            end
        end
    end
    points_to_fils()

    # A method to save the simulation to a file
    function save_filaments(step)
        p = zeros(3, total_points)
        cells = Vector{WriteVTK.MeshCell}(undef, total_fils)
        for i = 1 : total_points
            p[:, i] = points[i, :]
        end
        for i = 1 : total_fils
            e1 = i
            if i % f_per_ring == 0
                e2 = i - f_per_ring + 1
            else
                e2 = i + 1
            end
            cells[i] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_LINE, 
                [e1, e2])
        end
        vtkfile = WriteVTK.vtk_grid("filaments"*string(step), p, cells)
        WriteVTK.vtk_save(vtkfile)
    end

    save_filaments(0)
    for i = 1 : n_steps
        vels = filament_induced_velocity(fil_starts, fil_ends, fil_strs, points)
        points .+= dt .* vels
        points_to_fils()

        if i % 1 == 0
            save_filaments(i)
        end
    end
end
