using ProgressMeter, FFTW, Plots, LinearAlgebra, Dierckx
using BenchmarkTools, Statistics

function landau( epsilon, kx, x, v )
    
    (1.0.+epsilon*cos.(kx*x))/sqrt(2*pi) .* transpose(exp.(-0.5*v.*v))
    
end

function compute_rho(v, f)
    delta_v = v[2] - v[1]
    rho = delta_v .* vec(sum(real(f), dims=2))
    rho .- mean(rho)
end


function compute_E(x, rho, kx)
    Nx = length(x)
    kx_temp = copy(kx)
    kx_temp[1] = 1.0
    rho_hat = fft(rho)./kx_temp
    
    vec(real(ifft(-1im .* rho_hat)))
end


# function advection_step_v(f_t, E, v, kv, delta_t)

#     fft!(f_t, 1)
#     f_t .= f_t .* exp.(-1im * delta_t * kv * transpose(E))
#     ifft!(f_t, 1)
# end

function advection_step_v(f, E, v, delta_t)
    for i = 1:length(v)
        f_old_recons = Spline1D(v, real(f[i, :]), k=3)
        f[i, :] = f_old_recons(v .- E[i]*delta_t)
    end
end

function advection_step_x(f, E, v, kx, delta_t)
    Ev = exp.(-1im * delta_t * kx * transpose(v))

    fft!(f, 1)
    f .= f .* Ev
    delta_v = v[2] - v[1]
    rho = delta_v * vec(sum(f, dims=2))
    E[2:end] = -1im * rho[2:end] ./ kx[2:end]
    # for i in 2:length(E)
    #     E[i] = -1im * rho[i] ./ kx[i]
    # end
    
    # for i in 2:length(E)
    #     E[i] =
    E[1] = 0.0
    ifft!(f, 1)
    ifft!(E)
    E .= real(E)
end



function frequency_grid(grid, grid_max, grid_min, N)
    scaling_factor = 2*pi/(grid_max - grid_min)
    k = zeros(Float64, N)
    k .= scaling_factor .* [0:(N/2)-1; -(N/2):-1]
    return k
end


function vlasov_ampere(Nx, Nv, x_min, x_max, v_min, v_max, t_max, Nt)
    #delta_x = (x_max - x_min)/Nx
    #delta_v = (v_max - v_min)/Nv
    
    #get grid points
    x = range(x_min, x_max, Nx+1)[1:end-1]
    v = range(v_min, v_max, Nv+1)[1:end-1]
    kx = frequency_grid(x, x_max, x_min, Nx)
    kv = frequency_grid(v, v_max, v_min, Nv)
    
    delta_x = x[2] - x[1]
    delta_v = v[2] - v[1]
                             

    #allocate f, and f_t 
    f = zeros(Complex{Float64}, (Nx, Nv))
    f_t = zeros(Complex{Float64}, (Nv, Nx))

    #parameters for landau damping
    epsilon = 0.001
    k = 0.5

    #initialize f and f_t
    f .= landau(epsilon, k, x, v)
    transpose!(f_t, f)

    #initialize electric field
    rho = compute_rho(v, f)
    E = zeros(Complex{Float64}, Nx)
    E .= compute_E(x, rho, kx)

    Energy = Float64[]

    delta_t = t_max/Nt

    #do the advections
    for i in 1:Nt
        advection_step_v(f, E, v, 0.5*delta_t)
        # advection_step_v(f_t, E, v, kv, 0.5*delta_t)
        # transpose!(f, f_t)

        advection_step_x(f, E, v, kx, delta_t)

        #store calculated (electrical) energy
        push!(Energy, log(sqrt((sum(E.^2))*delta_x)))

        advection_step_v(f, E, v, 0.5*delta_t)

        # transpose!(f_t, f)
        # advection_step_v(f_t, E, v, kv, 0.5*delta_t)
    end
    real(f), Energy, E 
end








Nx, Nv = 256, 256
x_min, x_max = 0.0, 4*pi
v_min, v_max = -6.0, 6.0
t_max = 60
Nt = 600

#test_components(Nx, Nv, x_min, x_max, v_min, v_max, t_max, Nt)

t = range(0, stop=t_max, length=Nt)
EE_plot = plot(t, -0.1533*t.-5.48)
f, Energy, E = vlasov_ampere(Nx, Nv, x_min, x_max, v_min, v_max, t_max, Nt)
plot!(EE_plot, t, Energy, label="ampere",
      title="LLD(k, eps.) = (0.5, 0.001)",
      xlabel="time",
      ylabel="E Energy (log)",
      markershape=:x)

savefig(EE_plot, "VAmp1DSL.pdf")

gui()
