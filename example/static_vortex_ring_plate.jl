
push!(LOAD_PATH, "src/")
import WriteVTK
import cvortex

println("Running...")
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
