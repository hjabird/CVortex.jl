##############################################################################
#
# unsteady_vortex_lattice_method.jl
#
# An implementation of the UVLM. No force calculations or anything so neat.
#
# Copyright HJA Bird 2019
#
##############################################################################

using CVortex
using WriteVTK
using ForwardDiff

let
    aoa = deg2rad(0)
    # Define a rectangular rectangular wing
    wingshape = (x, t)->[x[1]/2 + t, x[2]*2, x[1]*sin(aoa) + 0.25*cos(t)]
    wing_xdisc = collect(-1:0.2:1)
    wing_ydisc = vcat(collect(-1:0.02: -0.8), collect(-0.6:0.2:0.6), collect(0.8:0.02:1))
    # Timestep
    dt = 0.1
    steps = 500

    # Lets try and make life neater through the use of a structure.
    # Only true academics are allowed use arrays and reshape for everything.
    mutable struct VortexLattice
        points :: Array{Float64, 3} # Idx i, idx j, idx point3d
        strengths :: Matrix{Float64}
    end
    # Generate a representation of our vortex lattice as required for
    # input to CVortex methods.
    function to_filaments(a::VortexLattice)
        @assert(size(a.points)[1]==size(a.strengths)[1]+1)
        @assert(size(a.points)[2]==size(a.strengths)[2]+1)
        i_max = size(a.points)[1]
        j_max = size(a.points)[2]
        local n_fils = i_max*(j_max-1) + (i_max-1)*j_max
        local fs_points = zeros(n_fils, 3)
        local fe_points = zeros(n_fils, 3)
        local f_strs = zeros(n_fils)
        # Filaments in the i direction.
        acc = 1
        for j = 1 : j_max
            for i = 1 : i_max - 1
                fs_points[acc, :] = a.points[i, j, :]
                fe_points[acc, :] = a.points[i+1, j, :]
                if j < j_max
                    f_strs[acc] += a.strengths[i, j]
                end
                if j > 1
                    f_strs[acc] -= a.strengths[i, j-1]
                end
                acc += 1
            end
        end
        # Filaments in the j direction
        for j = 1 : j_max - 1
            for i = 1 : i_max
                fs_points[acc, :] = a.points[i, j, :]
                fe_points[acc, :] = a.points[i, j+1, :]
                if i < i_max
                    f_strs[acc] -= a.strengths[i, j]
                end
                if i > 1
                    f_strs[acc] += a.strengths[i-1, j]
                end
                acc += 1
            end
        end
        return fs_points, fe_points, f_strs
    end
    # Generate a set of points in the matrix form required by CVortex
    function to_points(a::VortexLattice)
        local points = reshape(a.points, Int64(prod(size(a.points))/3), 3)
        return points
    end
	# 2D indexing to linear indexing (must match reshape)
    function point_lindex(a::VortexLattice, i::Int, j::Int)
        @assert(i > 0)
        @assert(j > 0)
        @assert(i <= size(a.points)[1])
        @assert(j <= size(a.points)[2])
        return (j-1)*size(a.points)[1] + i
    end
	# Linear index to 2D index (must match reshape)
    function str_index(a::VortexLattice, idx::Int)
        @assert(idx > 0)
        @assert(idx <= Int64(prod(size(a.strengths))))
        i = (idx-1)%size(a.strengths)[1] + 1
        j = Int64(floor((idx-1) / size(a.strengths)[1])) + 1
        return i, j
    end
    # Generate a set of cell centre points
    function to_cell_centre_points(a::VortexLattice)
        local points = zeros(prod(size(a.strengths)), 3)
        local t_points = (a.points[1:end-1,1:end-1, :] +
            a.points[1:end-1, 2:end, :] + a.points[2:end, 2:end, :] +
            a.points[2:end, 1:end-1, :]) ./ 4
        points = reshape(t_points, prod(size(a.strengths)), 3)
        return points
    end
    # Generate a set of cell centre normals
    function to_cell_centre_normals(a::VortexLattice)
        local normals = zeros(size(a.strengths)[1], size(a.strengths)[2], 3)
        for i = 1 : size(a.strengths)[1]
            for j = 1 : size(a.strengths)[2]
                p1 = a.points[i, j, :]
                p2 = a.points[i+1, j, :]
                p3 = a.points[i+1, j+1, :]
                p4 = a.points[i, j+1, :]
                t1 = (p4 + p3 - p2 - p1) ./ 4
                t2 = (p2 + p3 - p1 - p4) ./ 4
                normals[i, j, :] = [ t1[2]*t2[3]-t1[3]*t2[2],
                                    t1[3]*t2[1]-t1[1]*t2[3],
                                    t1[1]*t2[2]-t1[2]*t2[1]]
            end
        end
        return reshape(normals, Int64(length(normals)/3), 3)
    end
    # A function to make a VortexLattice with predefined geometry
    function make_VortexLattice(a_fn::Function,
        xinpt::Vector{<:Real}, yinpt::Vector{<:Real}, time::Real)
        local i_sz = length(xinpt)
        local j_sz = length(yinpt)
        local vl = VortexLattice(zeros(i_sz, j_sz, 3), zeros(i_sz-1, j_sz-1))
        for i = 1 : i_sz
            for j = 1 : j_sz
                vl.points[i, j, :] = a_fn([xinpt[i], yinpt[j]], time)
            end
        end
        return vl
    end
    # A function to get a sparse matrix transferring the filament influence
    # matrix to the ring influence matrix.
    function fil_to_ring_transform_mtrx(a::VortexLattice)
        @assert(size(a.points)[1]==size(a.strengths)[1]+1)
        @assert(size(a.points)[2]==size(a.strengths)[2]+1)
        i_max = size(a.points)[1]
        j_max = size(a.points)[2]
        local n_fils = i_max*(j_max-1) + (i_max-1)*j_max
        local n_strs = prod(size(a.strengths))
        local mtrx = zeros(n_fils, n_strs)
        # Filaments in the i direction.
        acc = 1
        for j = 1 : j_max
            for i = 1 : i_max - 1
                if j < j_max
                    mtrx[acc, (j-1) * (i_max-1) + i] += 1
                end
                if j > 1
                    mtrx[acc, (j-2) * (i_max-1) + i] -= 1
                end
                acc += 1
            end
        end
        # Filaments in the j direction
        for j = 1 : j_max - 1
            for i = 1 : i_max
                if i < i_max
                    mtrx[acc, (j-1) * (i_max-1) + i] -= 1
                end
                if i > 1
                    mtrx[acc, (j-1) * (i_max-1) + i-1] += 1
                end
                acc += 1
            end
        end
        return mtrx
    end
    # And we want to be able to make a pretty image:
    function save_mesh(i, lattices::Vector{VortexLattice})
        local total_cells = mapreduce(x->length(x.strengths), +, lattices; init=0)
        local total_points = mapreduce(x->Int64(prod(size(x.points))/3), +, lattices; init=0)
        local cells = Vector{WriteVTK.MeshCell}(undef, total_cells)
        local cell_strs = Vector{Float64}(undef, total_cells)
        local points = Matrix{Float64}(undef, total_points, 3)
        local poffset = 1
        local acc  = 1
        for lattice in lattices
            local np = Int64(prod(size(lattice.points))/3)
            local nc = length(lattice.strengths)
            points[poffset:poffset+np-1, :] = to_points(lattice)
            for il = 1 : nc
                ii, jj = str_index(lattice, il)
                p1 = point_lindex(lattice, ii, jj)
                p2 = point_lindex(lattice, ii+1, jj)
                p3 = point_lindex(lattice, ii+1, jj+1)
                p4 = point_lindex(lattice, ii, jj+1)
                cells[acc] = WriteVTK.MeshCell(
                    WriteVTK.VTKCellTypes.VTK_QUAD, (poffset-1) .+ [p1, p2, p3, p4])
                cell_strs[acc] = lattice.strengths[ii, jj]
                acc += 1
            end
            poffset += np
        end
        vtkfile = WriteVTK.vtk_grid("VLM_rings"*string(i), 
            points', cells)
        WriteVTK.vtk_cell_data(vtkfile, cell_strs, "Ring_strengths")
        WriteVTK.vtk_save(vtkfile)
    end

    # Now the meat of the code!
    wing = make_VortexLattice(wingshape, wing_xdisc, wing_ydisc, -dt)
    wake = VortexLattice(wing.points[1:1, :, :], zeros(0, length(wing_ydisc)-1))
	# This allows us to cheat a little later.
    wx_cc = (wing_xdisc[1:end-1] + wing_xdisc[2:end])/2
    wy_cc = (wing_ydisc[1:end-1] + wing_ydisc[2:end])/2

    for i = 1 : steps
		current_time = (i-1) * dt
		println("Step ",i)
		
        wing = make_VortexLattice(
            wingshape, wing_xdisc, wing_ydisc, current_time)
        buffer = VortexLattice(
            cat(wing.points[1:1, :, :], wake.points[1:1, :, :]; dims=1),
            zeros(1, length(wing_ydisc)-1))
			
        # First, compute the influence of the wake and kinematics on the wing
        wpoints = to_cell_centre_points(wing)
        wnormals = to_cell_centre_normals(wing)
        wake_fs, wake_fe, wake_str = to_filaments(wake)
        wake_indvel = filament_induced_velocity(wake_fs, wake_fe, wake_str,
            wpoints)
        wake_normal_inf = mapreduce(i->sum(wake_indvel[i,:].*wnormals[i,:]),
            vcat, 1:size(wnormals)[1])
        kinematic_indvel = map(
            x->ForwardDiff.derivative(t->wingshape(x, t), current_time),
            [(wx_cc[i], wy_cc[j]) for j in 1:length(wy_cc), i in 1:length(wx_cc)])
        kinematic_indvel = mapreduce(x->x, hcat, reshape(kinematic_indvel, prod(size(kinematic_indvel))))'
        kinematic_normal_inf = mapreduce(
            i->sum(kinematic_indvel[i,:].*wnormals[i,:]), vcat, 1:size(wnormals)[1])

        # Now, compute the influence matrix of the wing on the wing
        wing_fs, wing_fe, ~ = to_filaments(wing)
        buff_fs, buff_fe, ~ = to_filaments(buffer)
        wing_inf_mtrx = filament_induced_velocity_influence_matrix(
            wing_fs, wing_fe, wpoints, wnormals)
        buff_inf_mtrx = filament_induced_velocity_influence_matrix(
            buff_fs, buff_fe, wpoints, wnormals)
            
        wing_inf_mtrx = wing_inf_mtrx * fil_to_ring_transform_mtrx(wing)
        buff_inf_mtrx = buff_inf_mtrx * fil_to_ring_transform_mtrx(buffer)
        wing_inf_mtrx[:, 
            length(wing_xdisc)-1 : length(wing_xdisc)-1 : (length(wing_xdisc)-1)*(length(wing_ydisc)-1)] -= buff_inf_mtrx

        # And solve to get wing vorticity
        ring_str = wing_inf_mtrx \ (kinematic_normal_inf - wake_normal_inf)
        @assert(all(isfinite.(ring_str)))
        wing.strengths = reshape(ring_str, size(wing.strengths)[1], size(wing.strengths)[2])
        buffer.strengths = map(i->ring_str[(i[2]-1)*size(wing.strengths)[1]+i[1]], [(i, j) for i in size(wing.strengths)[1]:size(wing.strengths)[1], j in 1:size(wing.strengths)[2]])

        # Finally, compute the influence of the everything on the wake + buffer.
        wing_fs, wing_fe, wing_strs = to_filaments(wing)
        buff_fs, buff_fe, buff_strs = to_filaments(buffer)
        wake_fs, wake_fe, wake_strs = to_filaments(wake)
        buff_pts = to_points(buffer)
        wake_pts = to_points(wake)
        pts_vels = filament_induced_velocity(
            vcat(wing_fs, buff_fs, wake_fs), vcat(wing_fe, buff_fe, wake_fe),
            vcat(wing_strs, buff_strs, wake_str), vcat(buff_pts, wake_pts))
        buff_pts = reshape(buff_pts + dt * pts_vels[1:size(buff_pts)[1], :], size(buffer.points)[1], size(buffer.points)[2], 3)
        wake_pts = reshape(wake_pts + dt * pts_vels[end - size(wake_pts)[1]+1:end, :], size(wake.points)[1], size(wake.points)[2], 3)
        wake = VortexLattice(
            cat(buff_pts[1:1,:,:], wake_pts; dims=1),
            cat(buffer.strengths, wake.strengths; dims=1))
        save_mesh(i, [wing, buffer, wake])
    end

end #let
