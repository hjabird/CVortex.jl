
push!(LOAD_PATH, "src/")
import WriteVTK
import cvortex

println("Running...")
print("Disabling accelerator " * cvortex.cvortex_accelerator_name(0) * "\n")
# cvortex.cvortex_accelerator_disable(0)
let
    n_rings = 800
    p_per_ring = 150
    f_per_ring = p_per_ring
    total_points = n_rings * p_per_ring
    total_fils = n_rings * f_per_ring
    radius = 0.5
    length = 1
    str = 0.0005

    n_steps = 40
    dt = 0.1
    println("Total number of filaments is ", total_fils, ".")

    # Create a load of vortex particles in rings.
    points = Vector{cvortex.Vec3f}(undef, total_points)
    fils = Vector{cvortex.VortexFilament}(undef, total_fils)
    acc = 1
    for ring_idx = 1 : n_rings
        for p_in_ring_idx = 1 : p_per_ring
            xloc = ring_idx * length / n_rings
            yloc = radius * cos(2 * pi * p_in_ring_idx / p_per_ring)
            zloc = radius * sin(2 * pi * p_in_ring_idx / p_per_ring)
            points[acc] = cvortex.Vec3f(xloc, yloc, zloc)
            acc += 1
        end
    end

    function points_to_fils()
        local acc = 1
        for ring_idx = 1 : n_rings
            for p_in_ring_idx = 1 : f_per_ring
                e1 = points[acc]
                if p_in_ring_idx == f_per_ring
                    e2 = points[acc - f_per_ring + 1]
                else
                    e2 = points[acc + 1]
                end
                fils[acc] = cvortex.VortexFilament(e1, e2, str)
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
            p[1, i] = points[i].x
            p[2, i] = points[i].y
            p[3, i] = points[i].z
        end
        for i = 1 : total_fils
            e1 = i
            if i % f_per_ring == 0
                e2 = i - f_per_ring + 1
            else
                e2 = i + 1
            end
            cells[i] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_LINE, [e1, e2])
        end
        vtkfile = WriteVTK.vtk_grid("filaments"*string(step), p, cells)
        WriteVTK.vtk_save(vtkfile)
    end

    save_filaments(0)
    for i = 1 : n_steps
        println("Step "*string(i)*".")
        vels = cvortex.induced_velocity(fils, points)
        for i = 1 : total_points
            points[i] = cvortex.Vec3f(
                points[i].x + dt * vels[i].x,
                points[i].y + dt * vels[i].y,
                points[i].z + dt * vels[i].z,
            )
        end
        points_to_fils()

        if i % 1 == 0
            save_filaments(i)
        end
    end
end
