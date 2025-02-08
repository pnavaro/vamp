using Plots

include("vlasov-ampere2D.jl")

FFTW.set_num_threads(Threads.nthreads())

k_x1 = 0.4
k_x2 = 0.4

Nx1 = 32
Nx2 = 32
Nv1 = 64
Nv2 = 64
x1_min = 0.0
x2_min = 0.0
x1_max = 4.0*pi/k_x1
x2_max = 4.0*pi/k_x2
v1_min = -6.0
v2_min = -6.0
v1_max = 6.0
v2_max = 6.0
epsilon = 0.01

delta_t = 0.1
t_steps = 200
t_max = t_steps * delta_t
t = range(0, stop=t_max, length=t_steps + 1)




meshX = UniformMesh2D(x1_min, x1_max,
                      x2_min, x2_max,
                      Nx1, Nx2)
meshV = UniformMesh2D(v1_min, v1_max,
                      v2_min, v2_max,
                      Nv1, Nv2)


f = zeros(Complex{Float64}, meshX.Nx1, meshX.Nx2,
              meshV.Nx1, meshV.Nx2)
    
 
initLandau2D(f, epsilon, k_x1, k_x2, meshX, meshV)

rho = initRho(meshX, meshV, f)
phi_hat = initPhiHat(meshX, rho)

E = initE(meshX, phi_hat)
delta_t = t_max/t_steps
#@time EE, ET = vlasovAmpereSolveSLSplit(f, E, meshX, meshV, delta_t, t_steps)
@time EE, ET = vlasovAmpereSolveSL(f, E, meshX, meshV, delta_t, t_steps)
#@time EE, ET = vlasovAmpereSolveFourier(f, E, meshX, meshV, delta_t, t_steps)
E0 = ET[1]
ETrel = abs.((ET .- E0))

EE_plt = plot(t, EE./2.0,
              yaxis=:log,
              title="LLD(k_x1, k_x2, eps.) = ($(k_x1), $(k_x2), $(epsilon))",
              xlabel="time",
              ylabel="E. Energy",
              markershape=:x)
      
savefig(EE_plt, "Landau2DEE.pdf")
ETrel_plt = plot(t[2:end], ETrel[2:end],
                 yaxis=:log,
                 title="LLD(k_x1, k_x2, eps.) = ($(k_x1), $(k_x2), $(epsilon))",
                 xlabel="time",
                 ylabel="Total Energy Err.",
                 markershape=:x)


savefig(ETrel_plt, "Landau2DET.pdf")






              

