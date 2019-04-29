
push!(LOAD_PATH, "src/")
import cvortex

struct VortexRingSheet
    points_x :: Matrix{Float64}
    points_y :: Matrix{Float64}
    points_z :: Matrix{Float64}
    strengths :: Matrix{Float64}


    function VortexRingSheet(i::Int, j::Int, func::Function)
        @assert(hasmethod(j, (Float64, Float64)))
        @assert(i >= 0)
        @assert(j >= 0)

        px = Matrix{Float64}(undef, i+1, j+1)
        py = Matrix{Float64}(undef, i+1, j+1)
        pz = Matrix{Float64}(undef, i+1, j+1)
        str = zeros(i+1, j+1)
        for ii = 0:i
            for jj = 0: j
                coord = func(2 * ii / i - 1, 2 * jj / j - 1)
                px[ii+1, jj+1] = coord[1]
                py[ii+1, jj+1] = coord[2]
                pz[ii+1, jj+1] = coord[3]
                str = 0
            end
        end
        new(px, py, pz, str)
    end
end

function get_node_points(a::VortexRingSheet)
    @assert(size(a.points_x) == size(a.points_y))
    @assert(size(a.points_x) == size(a.points_z))

    np = size(a.points_y)[1] * size(a.points_y)[2]
    points = Matrix{cvortex.Vec3f}(undef, size(a.points_y))
    for i = 1 : size(a.points_y)[1]
        for j = 1 : size(a.points_y)[2]
            points[i, j] = cvortex.Vec3f(a.points_x[i,j], a.points_y[i, j], a.points_z[i, j])
        end
    end
    return vec(points)
end

function apply_velocity(a::VortexRingSheet, vel::Vector{cvortex.Vec3f}, dt :: Real)
    @assert(length(vel) == length(a.points_x))
    vel = reshape(vel, size(a.points_x))
    a.points_x += map(x->x.x, vel) .* dt
    a.points_y += map(x->x.y, vel) .* dt
    a.points_z += map(x->x.z, vel) .* dt
    return
end

function get_ring_centres(a::VortexRingSheet)
    sz = size(a.str)
    @assert(sz[1] == size(a.points_x)[1]-1)
    @assert(sz[2] == size(a.points_x)[2]-1)
    cx = Matrix{Float64}(undef, sz)
    cy = Matrix{Float64}(undef, sz)
    cz = Matrix{Float64}(undef, sz)
    
    for i = 1 : sz[1]
        for j = 1 : sz[2]
            cx[i, j] = (a.points_x[i, j+1] + a.points_x[i+1, j+1] + a.points_x[i+1, j] + a.points_x[i, j])/4
            cy[i, j] = (a.points_y[i, j+1] + a.points_y[i+1, j+1] + a.points_y[i+1, j] + a.points_y[i, j])/4
            cz[i, j] = (a.points_z[i, j+1] + a.points_z[i+1, j+1] + a.points_z[i+1, j] + a.points_z[i, j])/4
        end
    end
    return cx, cy, cz
end

function get_ring_centres(a::VortexRingSheet)
    sz = size(a.str)
    @assert(sz[1] == size(a.points_x)[1]-1)
    @assert(sz[2] == size(a.points_x)[2]-1)
    cx = Matrix{Float64}(undef, sz)
    cy = Matrix{Float64}(undef, sz)
    cz = Matrix{Float64}(undef, sz)
    
    for i = 1 : sz[1]
        for j = 1 : sz[2]
            cx[i, j] = (a.points_x[i, j+1] + a.points_x[i+1, j+1] + a.points_x[i+1, j] + a.points_x[i, j])/4
            cy[i, j] = (a.points_y[i, j+1] + a.points_y[i+1, j+1] + a.points_y[i+1, j] + a.points_y[i, j])/4
            cz[i, j] = (a.points_z[i, j+1] + a.points_z[i+1, j+1] + a.points_z[i+1, j] + a.points_z[i, j])/4
        end
    end
    return cx, cy, cz
end

function get_ring_normals(a::VortexRingSheet)
    sz = size(a.str)
    @assert(sz[1] == size(a.points_x)[1]-1)
    @assert(sz[2] == size(a.points_x)[2]-1)
    nx = Matrix{Float64}(undef, sz)
    ny = Matrix{Float64}(undef, sz)
    nz = Matrix{Float64}(undef, sz)
    
    for i = 1 : sz[1]
        for j = 1 : sz[2]
            # Bilinear shape function derivatives
            dfxdx = -a.points_x[i, j]/4 + a.points_x[i, j+1]/4 + a.points_x[i+1, j+1]/4 - a.points_x[i+1, j]/4
            dfydx = -a.points_y[i, j]/4 + a.points_y[i, j+1]/4 + a.points_y[i+1, j+1]/4 - a.points_y[i+1, j]/4
            dfzdx = -a.points_z[i, j]/4 + a.points_z[i, j+1]/4 + a.points_z[i+1, j+1]/4 - a.points_z[i+1, j]/4
            dfxdy = -a.points_x[i, j]/4 - a.points_x[i, j+1]/4 + a.points_x[i+1, j+1]/4 + a.points_x[i+1, j]/4
            dfydy = -a.points_y[i, j]/4 - a.points_y[i, j+1]/4 + a.points_y[i+1, j+1]/4 + a.points_y[i+1, j]/4
            dfzdy = -a.points_z[i, j]/4 - a.points_z[i, j+1]/4 + a.points_z[i+1, j+1]/4 + a.points_z[i+1, j]/4
            # Cross products
            nx[i, j] = dfydx * dfzdy - dfzdx * dfydy
            ny[i, j] = dfzdx * dfxdy - dfxdx * dfzdy
            nz[i, j] = dfxdx * dfydy - dfydx * dfxdy
        end
    end
    return nx, ny, nz
end

function induced_velocity(a::VortexRingSheet, mes_pts::Matrix{T})
    @assert(size(mes_pts)[2] == 3)
    szr = size(a.str)
    nfil = szr[1] * szr[2] * 2 + szr[1] + szr[2]
    # Prep Filaments
    fils = Vector{cvortex.VortexFilament}(undef, nfil)
    nxx = szr[2]
    nxy = szr[2] + 1
    nyx = szr[1] + 1
    nyy = szr[1]
    for i = 1 : nxx * nxy # Lets do y filaments first.

    end


end

function to_filaments(a::VortexRingSheet)
    szr = size(a.str)
    nfil = szr[1] * szr[2] * 2 + szr[1] + szr[2]
    fils = Vector{cvortex.VortexFilament}(undef, nfil)
    nxx = szr[2]
    nxy = szr[2] + 1
    nyx = szr[1] + 1
    nyy = szr[1]
    for i = 0 : nxx * nxy - 1# Lets do y filaments first.
        ii = i%(srz[1]) + 1
        jj = Int64(ceil(i/(srz[i])))
        coord1 = cvortex.Vec3f(a.points_x[])
    end

end

let
    xs = collect(-1:1:1);
    ys = collect(-1:2/3:1);
    xmes = (xs[1:end-1] + xs[2:end]) / 2
    ymes = (ys[1:end-1] + ys[2:end]) / 2
    zpos = 0.0

    fils = Vector{cvortex.VortexFilament}(undef, length(xs)*length(ymes) + length(ys)*length(xmes))
    for i = 0 : length(xs)*length(ymes)-1 # Filaments in y direction. 
        yidx1 = Int64(floor(i/length(xs))) + 1
        yidx2 = yidx1 + 1
        xidx = i%length(xs) + 1
        fils[i+1] = cvortex.VortexFilament(
            cvortex.Vec3f(xs[xidx], ys[yidx1], zpos),
            cvortex.Vec3f(xs[xidx], ys[yidx2], zpos),
            1.0
        )
    end    
    xoffset = length(xs)*length(ymes)
    for i = 0 : length(ys)*length(xmes)-1 # Filaments in x direction. 
        yidx = Int64(floor(i/length(xmes))) + 1
        xidx1 = i%length(xmes) + 1
        xidx2 = xidx1 + 1
        fils[i + xoffset + 1] = cvortex.VortexFilament(
            cvortex.Vec3f(xs[xidx1], ys[yidx], zpos),
            cvortex.Vec3f(xs[xidx2], ys[yidx], zpos),
            1.0
        )
    end
    for i = 1: length(fils)
        fil = fils[i]
        println(i, ": ", fil.coord1, " ", fil.coord2)
    end
    # Make measurement points...
    mes_pts = Vector{cvortex.Vec3f}(undef, length(xmes) * length(ymes))
    mes_dirs = Vector{cvortex.Vec3f}(undef, length(xmes) * length(ymes))
    for i = 1 : length(xmes)
        for j = 1 : length(ymes)
            mes_pts[(i-1)*length(ymes) + j] = cvortex.Vec3f(xmes[i], ymes[j], zpos)
            mes_dirs[(i-1)*length(ymes) + j] = cvortex.Vec3f(1.0, 1.0, 1.0)
        end
    end
    # Now we can compute the influence matrix:
    influence = cvortex.induced_velocity_influence_matrix(fils, mes_pts, mes_dirs)
    println("Filament influence matrix. Transforming to Ring influence.")
    # Compute the ring influences
    ring_strengths = Vector{Float64}(undef, length(xmes)*length(ymes))
    ring_influence = Matrix{Float64}(undef, length(xmes)*length(ymes), length(xmes)*length(ymes))
    for j  = 1 : length(xmes)*length(ymes) #
        xpidx = xoffset + j
        xmidx = xpidx + length(xmes)
        ypidx = j + Int64(ceil(j/length(xmes)))
        ymidx = ypidx - 1
        println("xp: ", xpidx, " :", fils[xpidx].coord1, fils[xpidx].coord2)
        println("xm: ", xmidx, " :", fils[xmidx].coord1, fils[xmidx].coord2)
        println("yp: ", ypidx, " :", fils[ypidx].coord1, fils[ypidx].coord2)
        println("ym: ", ymidx, " :", fils[ymidx].coord1, fils[ymidx].coord2)
        ring_influence[:, j] = influence[:, xpidx] - influence[:, xmidx] + influence[:, ypidx] - influence[:, ymidx]
    end
    ring_strengths = ring_influence \ ones(Float64, length(xmes)*length(ymes))


end
