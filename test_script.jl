include("tests.jl")



Nx1 = 32
Nx2 = 32
Nv1 = 64
Nv2 = 64
x1_min = 0.0
x2_min = 0.0
x1_max = 4.0*pi
x2_max = 4.0*pi
v1_min = -6.0
v2_min = -6.0
v1_max = 6.0
v2_max = 6.0
epsilon = 0.005
k_x1 = 0.5
k_x2 = 0.5

t_steps = 300
t_max = t_steps * 0.1
t = range(0, stop=t_max, length=t_steps)




meshX = UniformMesh2D(x1_min, x1_max,
                      x2_min, x2_max,
                      Nx1, Nx2)
meshV = UniformMesh2D(v1_min, v1_max,
                      v2_min, v2_max,
                      Nv1, Nv2)


f = zeros(Complex{Float64}, meshX.Nx1, meshX.Nx2,
              meshV.Nx1, meshV.Nx2)
    
test_initLandau2D(f,
                   epsilon,
                   k_x1,
                   k_x2,
                   meshX,
                   meshV)

