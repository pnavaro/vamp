#Code for linear advection using cubic B-splines
using Dierckx, Plots

struct UniformMesh1D
    x_min :: Float64
    x_max :: Float64
    Nx :: Int64
    delta_x :: Float64
    x :: Vector{Float64}
    function UniformMesh1D(x_min, x_max, Nx)
        x = range(x_min, x_max, Nx + 1)[1:end-1]
        delta_x = (x_max - x_min)/Nx
        new(x_min, x_max, Nx, delta_x, x)
    end
end

function advection_sl_1D( f, A,
                          delta_t,
                          mesh_x)
    # Interpolate
    f_old_recons = Spline1D(mesh_x.x, f, k = 5, bc="extrapolate", periodic=true)
    eval_points = zeros(Float64, mesh_x.Nx)
    eval_points .= mesh_x.x .- (delta_t * A)
    f .= f_old_recons(eval_points)
end

function stepFunc( x )
    if ((x >= -0.5) && (x <= 0.5 ))
        return 1.0
    else
        return 0.0
    end
end


Nx = 200
delta_t = 0.01
x_min = -10.0
x_max = 10.0
A = 1.2
mesh = UniformMesh1D(x_min, x_max, Nx)
#f = sin.(mesh.x)
f = stepFunc.(mesh.x)

for t_steps = 1:350
    advection_sl_1D(f, A, delta_t, mesh)
end

plt = plot(mesh.x, f,
           xlabel="x",
           ylabel="f",
           title="dt=$(delta_t), Nx=$(Nx), A=$(A), Nt=$(350)",
           markershape=:x) 

plot!(plt, mesh.x, stepFunc.(mesh.x .- (1.2*3.5)))
      
savefig(plt, "1DAdvectionSL.pdf")

