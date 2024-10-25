using FFTW, Plots, Dierckx

struct UniformMesh2D
    x1_min :: Float64
    x1_max :: Float64
    x2_min :: Float64
    x2_max :: Float64
    Nx1 :: Int64
    Nx2 :: Int64
    delta_x1 :: Float64
    delta_x2 :: Float64
    x1 :: Vector{Float64}
    x2 :: Vector{Float64}
    k1 :: Vector{Float64}
    k2 :: Vector{Float64}
    scaling_factor1 :: Float64
    scaling_factor2 :: Float64

    function UniformMesh2D(x1_min, x1_max,
                           x2_min, x2_max,
                           Nx1, Nx2)
        x1 = range(x1_min, x1_max, Nx1+1)[1:end-1]
        x2 = range(x2_min, x2_max, Nx2+1)[1:end-1]
        scaling_factor1 = 2*pi/(x1_max - x1_min)
        scaling_factor2 = 2*pi/(x2_max - x2_min)
        delta_x1 = (x1_max - x1_min)/Nx1
        delta_x2 = (x2_max - x2_min)/Nx2
        k1 = scaling_factor1 * vcat(0:(Nx1/2 - 1), -(Nx1/2):-1)
        k2 = scaling_factor2 * vcat(0:(Nx2/2 - 1), -(Nx2/2):-1)
        Nx1 = Nx1
        Nx2 = Nx2

        new(x1_min,
            x1_max,
            x2_min,
            x2_max,
            Nx1,
            Nx2,
            delta_x1,
            delta_x2,
            x1,
            x2,
            k1,
            k2,
            scaling_factor1,
            scaling_factor2)
    end
end


function step2D(x1, x2)
    if ((-0.5 < x1)&&(x1 < 0.5) && (-0.5 < x2) && (x2 < 0.5))
        return 1.0
    else
        return 0.0
    end
end

function advect2DLinear(f, E, meshX, delta_t)
    f_old_k = fft(f, (1, 2))
    for i = meshX.Nx1, j = meshX.Nx2
            #E_dot_k = E[1] * meshX.k1[i] + E[2] * meshX.k2[j]
            #println("Edotk: $(E_dot_k)")
            f_old_k[i, j] = f_old_k[i, j] * exp(-1.0im * (E[1]*meshX.k1[i] + E[2]*meshX.k2[j]) * delta_t)
    end
    f .= ifft(f_old_k, (1, 2))
end

function advection2DLinearSpline(f, E, meshX, delta_t)
    f_old_recons = Spline2D(meshX.x1, meshX.x2, real(f), kx=3, ky=3)
    for i = 1:meshX.Nx1, j = 1:meshX.Nx2
            f[i, j]= f_old_recons(meshX.x1[i] .- E[1]*delta_t, meshX.x2[j] - E[2]*delta_t)
    end
    
end


Nx1 = 256
Nx2 = 256
delta_t = 0.01
x1_min = -4.0
x1_max = 4.0
x2_min = -4.0
x2_max = 4.0
Nt = 300
E = [2.0; 2.0]

meshX = UniformMesh2D(x1_min, x1_max, x2_min, x2_max, Nx1, Nx2)

f = zeros(Complex{Float64}, Nx1, Nx2)
for i = 1:Nx1
    for j = 1:Nx2
        f[i, j] = step2D(meshX.x1[i], meshX.x2[j])
    end
end
initf = copy(f)
initf2= copy(f)
for i = 1:Nt
    advect2DLinear(f, E, meshX, delta_t)
    #advection2DLinearSpline(f, E, meshX, delta_t)
end

f2 = fft(initf2, (1, 2))

for i = 1:meshX.Nx1, j = meshX.Nx2
    f2[i,j]=f2[i,j] * exp(-1.0im * (E[1] * meshX.k1[i] + E[2]*meshX.k2[j])* delta_t * Nt)
end
ifft!(f2, (1, 2))

#advect2DLinear(initf, E, meshX, 3.0)

plt = plot(meshX.x1, meshX.x2, real(f), st=:surface, camera=(-30, 30))
#plot!(plt, meshX.x1, meshX.x2, real(initf), st=:surface, camera=(-30, 30))
    
    
