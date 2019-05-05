##############################################################################
#
# benchmark_vortex_particles.jl
#
# Benchmark the number of interactions a system can model per second
# with and without acceleration.
#
##############################################################################

using CVortex
using Dates
using Plots
plotly()

let
    # We can print information about the system.
    println("Running benchmark_vortex_particles.jl")
    println("CVortex found ", number_of_accelerators(), " accelerators "
        *"on this system.")
    println("There are ", number_of_enabled_accelerators(), 
        " enabled accelerators. These are:")
    for i = 1 : number_of_enabled_accelerators()
        println(i, ":\t", accelerator_name(i))
    end

    # We'll only start new runs for a certain amount of time:
    maxtime = 5
    println("Maxtime is ", maxtime, " seconds. The program will finish
        what its working on when it runs out of time.")
    nparticles = sort(vcat(map(i->2^i, 6:30), [1023]))

    # And the regularisation of the particle-particle interaction
    kernel = winckelmans_regularisation()

    r_particles = Vector{Int64}(undef,0)
    r_times = Vector{Float64}(undef, 0)

    starttime = now()
    println("Running...")
    i = 1
    while Float64((now() - starttime).value) / 1000 < maxtime
        ninter = nparticles[i] ^2
        particle_pos = rand(nparticles[i], 3)
        particle_vorts = rand(nparticles[i], 3)
        tstart = now()
        vels = particle_induced_velocity(particle_pos, particle_vorts, 
            particle_pos, kernel, 0.01)
        dvorts = particle_induced_dvort(particle_pos, particle_vorts, 
            particle_pos, particle_vorts, kernel, 0.01)
        tend = now()
        wallclocktime = round(Float64((tend - tstart).value))/1000
        if wallclocktime > 0
            push!(r_particles, nparticles[i])
            push!(r_times, round(Float64((tend - tstart).value))/1000)
        end
        i += 1
    end
    println("Particles: \tTime(s):\tBandwidth:\n")
    map(i->println(
        r_particles[i],"\t",
        r_times[i],"\t",
        r_particles[i]^2/r_times[i]), 1:length(r_particles))
    #plot(r_particles, r_times, xlabel="Number of particles",
    #    ylabel="Time(s)", xaxis=:log, yaxis=:log, shape=:circle)
    return
end
