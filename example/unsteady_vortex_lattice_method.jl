
using CVortex
using WriteVTK
using ForwardDiff

let
    aoa = deg2rad(10)
    # Define a rectangular rectangular wing
    wingshape = (x, t)->[x[1]*0.5 + t, x[2]*2, x[1]*sin(aoa)]
    wing_xdisc = collect(-1:0.2:1)
    wing_ydisc = collect(-1:0.2:1)
    # Timestep
    dt = 0.05
    steps = 20

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
        local points = zeros(prod(size(a.points))/3, 3)
        acc = 1
        for i = 1 : size(a.points)[1]
            for j = 1 : size(a.points)[2]
                points[acc, :] = a.points[i, j, :]
                acc += 1
            end
        end
        return points
    end
    # Generate a set of cell centre points
    function to_cell_centre_points(a::VortexLattice)
        local points = zeros(prod(size(a.strengths)), 3)
        local t_points = (a.points[1:end-1,1:end-1, :] +
            a.points[1:end-1, 2:end, :] + a.points[2:end, 2:end, :] +
            a.points[2:end, 1:end-1, :]) ./ 4
        acc = 1
        for i = 1 : size(t_points)[1]
            for j = 1 : size(t_points)[2]
                points[acc, :] = a.points[i, j, :]
                acc += 1
            end
        end
        return points
    end
    # Generate a set of cell centre normals
    function to_cell_centre_normals(a::VortexLattice)
        local normals = zeros(prod(size(a.strengths)), 3)
        acc = 1
        for i = 1 : size(a.strengths)[1]
            for j = 1 : size(a.strengths)[2]
                p1 = a.points[i, j, :]
                p2 = a.points[i+1, j, :]
                p3 = a.points[i+1, j+1, :]
                p4 = a.points[i, j=1, :]
                t1 = (p4 + p3 - p2 - p1) ./ 4
                t2 = (p2 + p3 - p1 - p4) ./ 4
                normals[acc, :] = [ t1[2]*t2[3]-t1[3]*t2[2],
                                    t1[3]*t2[1]-t1[1]*t2[3],
                                    t1[1]*t2[2]-t1[2]*t2[1]]
                acc += 1
            end
        end
        return normals
    end
    # A function to make a VortexLattice with predefined geometry
    function make_VortexLattice(a_fn::Function,
        xinpt::Vector{<:Real}, yinpt::{<:Real}, time::Real)
        local i_sz = length(xinpt)
        local y_sz = length(yinpt)
        local vl = VortexLattice(zeros(i_sz, y_sz), zeros(i_sz-1, y_sz-1))
        for i = 1 : i_sz
            for j = 1 : j_sz
                a.points[i, j, :] = a_fn([x_inpt[i], y_inpt[j]], time)
            end
        end
        return
    end

    # Now the meat of the code!
    wing = make_VortexLattice(wingshape, wing_xdisc, wing_ydisc, time-dt)
    wake = VortexLattice(wing.points[end, :, :], zeros(0, length(wing_ydisc)))

    current_time = 0
    for i = 1 : steps
        wing = make_VortexLattice(
            wingshape, wing_xdisc, wing_ydisc, current_time)
        buffer = VortexLattice(
            cat(wing.points[end, :, :], wake[1, :, :]; dims=1),
            zeros(1, length(wing_ydisc)))

        # First, compute the influence of the wake and kinematics on the wing
        wpoints = to_cell_centre_points(wing)
        wnormals = to_cell_centre_normals(wing)
        wake_fs, wake_fe, wake_str = to_filaments(wake)
        wake_indvel = filament_induced_velocity(wake_fs, wake_fe, wake_str,
            wpoints)
        wake_normal_inf = mapreduce(i->sum(wake_indvel[i,:].*wnormals[i,:]),
            vcat(), 1:size(wnormals)[1])
        kinematics =

        # Now, compute the influence matrix of the wing wing on the wing
        # and solve to get wing vorticity
        wing_fs, wing_fe, ~ = to_filaments(wing)
        buff_fs, buff_fe, ~ = to_filaments(buffer)
        wing_inf_mtrx = filament_induced_velocity_influence_matrix(
            wing_fs, wing_fe, wpoints, wnormals)
        buff_inf_mtrx = filament_induced_velocity_influence_matrix(
            buff_fs, buff_fe, wpoints, wnormals)




        # Finally, compute the influence of the everything on the wake

    wake.points = cat(wing.points[end, :, :], wake.points;dims=1)
    wake.strengths = cat(wing.strengths[end, :], wake.strengths;dims=1)
    steps

end #let
