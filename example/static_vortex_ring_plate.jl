##############################################################################
#
# static_vortex_ring_plate.jl
#
# The most simple possible demo of using the vortex filament influence matrix
# function. No file output.
#
##############################################################################

using CVortex
import WriteVTK

println("Running static_vortex_ring_plate.jl")
let
    xs = collect(-1:.04:1);
    ys = collect(-1:.04:1);
    xmes = (xs[1:end-1] + xs[2:end]) / 2
    ymes = (ys[1:end-1] + ys[2:end]) / 2
    zpos = 0.0

    # Useful numbers to save us using length() and size()
    nf = length(xs)*length(ymes) + length(ys)*length(xmes)
    nc = length(xmes)*length(ymes)
    nxp = length(xs)
    nyp = length(ys)
    nxc = length(xmes)
    nyc = length(ymes)

    println("Generating ", nc, " vortex rings...")
    fil_fs = zeros(nf, 3)
    fil_fe = zeros(nf, 3)
    for i = 0 : length(xs)*length(ymes)-1 # Filaments in y direction. 
        yidx1 = Int64(floor(i/length(xs))) + 1
        yidx2 = yidx1 + 1
        xidx = i%length(xs) + 1
        fil_fs[i+1, :] = [xs[xidx], ys[yidx1], zpos]
        fil_fe[i+1, :] = [xs[xidx], ys[yidx2], zpos]
    end    
    xoffset = length(xs)*length(ymes)
    for i = 0 : length(ys)*length(xmes)-1 # Filaments in x direction. 
        yidx = Int64(floor(i/length(xmes))) + 1
        xidx1 = i%length(xmes) + 1
        xidx2 = xidx1 + 1
        fil_fs[i + xoffset + 1, :] = [xs[xidx1], ys[yidx], zpos]
        fil_fe[i + xoffset + 1, :] = [xs[xidx2], ys[yidx], zpos]
    end

    println("Generating ", nc, " measurement points and normals...")
    mes_pts = zeros(nc, 3)
    mes_dirs = zeros(nc, 3)
    for i = 1 : nxc
        for j = 1 : nyc
            mes_pts[(i-1)*nyc + j, :] = [xmes[i], ymes[j], zpos]
            mes_dirs[(i-1)*nyc + j, :] = [1.0, 1.0, 1.0]
        end
    end

    println("Computing self influence matrix (", nc, "x", nc, "=", nc^2,")...")
    influence = filament_induced_velocity_influence_matrix(fil_fs, fil_fe, mes_pts, mes_dirs)
    println("Filament influence matrix. Transforming to Ring influence.")
    # Compute the ring influences
    ring_strengths = Vector{Float64}(undef, nc)
    ring_influence = Matrix{Float64}(undef, nc, nc)
    for j  = 1 : nc
        xpidx = xoffset + j
        xmidx = xpidx + length(xmes)
        ypidx = j + Int64(ceil(j/length(xmes)))
        ymidx = ypidx - 1
        ring_influence[:, j] = influence[:, xpidx] - influence[:, xmidx] + influence[:, ypidx] - influence[:, ymidx]
    end
    println("Solving for ring vorticity given desired ring normal velocities...")
    ring_strengths = ring_influence \ ones(Float64, length(xmes)*length(ymes))

    println("Generating static_vortex_ring_plate.vtu...")
    cells = Vector{WriteVTK.MeshCell}(undef, nc)
    points = mapreduce(x->x, vcat, reshape([[xp yp 0] for xp in xs, yp in ys], nxp*nyp))
    strs = zeros(nc)
    acc = 1
    for i = 1 : nxc
        for j = 1 : nyc
            p1 = (j-1)*nxp + i
            p2 = (j)*nxp + i
            p3 = (j)*nxp + i + 1
            p4 = (j-1)*nxp + i + 1 
            cells[acc] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_QUAD, 
                [p1, p2, p3, p4])
            strs[acc] = ring_strengths[(j-1)*nxc + i]
            acc += 1
        end
    end
    vtkfile = WriteVTK.vtk_grid("static_vortex_ring_plate", 
        points', cells)
    WriteVTK.vtk_cell_data(vtkfile, strs, "Ring_strengths")
    WriteVTK.vtk_save(vtkfile)

    println("Done.")
end
