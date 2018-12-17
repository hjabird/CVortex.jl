include("cvortex.jl")
n = 1000;
pos = map(x->Vec3f(rand(), rand(), rand()), 1:n);
pos2  = map(x->Vec3f(rand(), rand(), rand()), 1:n);
vort = map(x->Vec3f(rand(), rand(), rand()), 1:n);
vort2 = map(x->Vec3f(rand(), rand(), rand()), 1:n);
particles = map(x->VortexParticle(pos[x], vort[x], 0.001), 1:n)

println("Vel 1-1")
induced_velocity(particles[1], pos[2], VortFunc_singular())
println("Vort 1-1")
induced_dvort(particles[1], particles[2], VortFunc_gaussian())

println("Vel n-1")
induced_velocity(particles, pos[1], VortFunc_planetary())
println("Vort n-1")
induced_dvort(particles, particles[1], VortFunc_planetary())

println("Vel n-m")
induced_velocity(particles, pos, VortFunc_planetary())
println("Vort n-m")
induced_dvort(particles, particles, VortFunc_planetary())
