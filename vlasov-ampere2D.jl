using FFTW, LinearAlgebra, Statistics, Dierckx
import Base.Threads.@threads

mutable struct UniformMesh2D

    #Physical space coordinates
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

    #Frequency space coordinates
    scaling_factor1 :: Float64
    scaling_factor2 :: Float64
    k1 :: Vector{Float64}
    k2 :: Vector{Float64}

    #Scratch space
    scratch :: Array{Complex{Float64}}

    function UniformMesh2D(x1_min, x1_max,
                           x2_min, x2_max,
                           Nx1 :: Int64, Nx2 :: Int64)

        #Initialize physical space coordinates
        x1 = range(x1_min, x1_max, Nx1 + 1)[1:end-1]
        x2 = range(x2_min, x2_max, Nx2 + 1)[1:end-1]
        
        delta_x1 = (x1_max - x1_min)/Nx1
        delta_x2 = (x2_max - x2_min)/Nx2

        #Initialize frequency space coordinates
        scaling_factor1 = 2*pi/(x1_max - x1_min)
        scaling_factor2 = 2*pi/(x2_max - x2_min)
        k1 = scaling_factor1 * vcat(0:(Nx1/2 - 1), -(Nx1/2):-1)
        k2 = scaling_factor2 * vcat(0:(Nx2/2 - 1), -(Nx2/2):-1)

        scratch = zeros(Complex{Float64}, Nx1*Nx2)

        new(x1_min, x1_max,
            x2_min, x2_max,
            Nx1, Nx2,
            delta_x1, delta_x2,
            x1, x2,
            scaling_factor1,
            scaling_factor2,
            k1, k2,
            scratch)
    end
end

struct ElectricField2D
    x1 :: Array{Float64, 2}
    x2 :: Array{Float64, 2}
    k1 :: Array{Complex{Float64}, 2}
    k2 :: Array{Complex{Float64}, 2}

    function ElectricField2D(Nx1 :: Int64, Nx2 :: Int64)
        x1 = zeros(Float64, Nx1, Nx2)
        x2 = zeros(Float64, Nx1, Nx2)
        k1 = zeros(Complex{Float64}, Nx1, Nx2)
        k2 = zeros(Complex{Float64}, Nx1, Nx2)
        new(x1, x2, k1, k2)
    end
end


function initLandau2D(f :: Array{Complex{Float64}, 4},
                       epsilon :: Float64,
                       k_x1 :: Float64,
                       k_x2 :: Float64,
                       meshX :: UniformMesh2D,
                       meshV :: UniformMesh2D)

    @threads for i = 1:length(meshV.x1)
        for j = 1:length(meshV.x2)
            coeff = exp(-0.5*( (meshV.x1[i]^2) + (meshV.x2[j]^2)))
            for k=1:length(meshX.x1)
                for l=1:length(meshX.x2)
                    f[k, l, i, j] = coeff * (1.0 + epsilon * cos(k_x1 * meshX.x1[k])*cos(k_x2 * meshX.x2[l]))
                end
            end
        end
    end
    f .= f./(2*pi)

end

function initRho(meshX :: UniformMesh2D,
                          meshV :: UniformMesh2D,
                          f :: Array{Complex{Float64}, 4})
    dV = meshV.delta_x1 * meshV.delta_x2
    rho = dV .* reshape(vec(sum(real(f), dims=(3,4))),
                        (meshX.Nx1, meshX.Nx2))
    #return rho .- mean(rho)
    return rho .- 1.0
end

function computeRho(rho :: Array{Complex{Float64}, 2},
                     meshX :: UniformMesh2D,
                     meshV :: UniformMesh2D,
                     f :: Array{Complex{Float64}, 4})
    dV = meshV.delta_x1 * meshV.delta_x2
    rho .= dV .* reshape(vec(sum(real(f), dims=(3,4))),
                         (meshX.Nx1, meshX.Nx2))
    #rho .= rho .- mean(rho)
    rho .= rho .- 1.0
    
end



function initPhiHat(meshX :: UniformMesh2D,
                              rho :: Array{Float64, 2})

    phi_hat = zeros(Complex{Float64}, meshX.Nx1, meshX.Nx2)
    mean_rho=mean(rho)
    rho_hat = fft(rho, (1, 2))
    phi_hat[1, 1] = 0.0
    @threads for i = 1:meshX.Nx1
        for j = 1:meshX.Nx2
            if ((i == 1) && (j == 1))
                phi_hat[i, j] = mean_rho
            else
                phi_hat[i, j] = 1.0/(meshX.k1[i]^2 + meshX.k2[j]^2)
            end
        end
    end
    
    phi_hat .= phi_hat .* rho_hat
    return phi_hat
end

function computePhiHat(phi_hat :: Array{Complex{Float64}, 2},
                         meshX :: UniformMesh2D,
                         rho :: Array{Complex{Float64}, 2})

    mean_rho = mean(rho)
    fft!(rho, (1, 2))
    
    phi_hat[1, 1] = 0.0
    @threads for i = 1:meshX.Nx1
        for j = 1:meshX.Nx2
            if ((i == 1) && (j == 1))
                phi_hat[i, j] = mean_rho
            else
                phi_hat[i, j] = 1.0/(meshX.k1[i]^2 + meshX.k2[j]^2)
            end
        end
    end
    
    phi_hat .= phi_hat .* rho
    
    return phi_hat
end



function initE(meshX :: UniformMesh2D,
               phi_hat :: Array{Complex{Float64}, 2})

    E = ElectricField2D(meshX.Nx1, meshX.Nx2)

    for i = 1:meshX.Nx1
        for j = 1:meshX.Nx2
            E.k1[i, j] = -1.0im * meshX.k1[i] * phi_hat[i, j]
            E.k2[i, j] = -1.0im * meshX.k2[j] * phi_hat[i, j]
        end
    end

    E.x1 .= real(ifft(E.k1, (2, 1)))
    E.x2 .= real(ifft(E.k2, (2, 1)))
    return E
end

function computeE(E :: ElectricField2D, 
                   meshX :: UniformMesh2D,
                   phi_hat :: Array{Complex{Float64}, 2})

    @threads for i = 1:meshX.Nx1
        for j = 1:meshX.Nx2
            E.k1[i, j] = -1.0im * meshX.k1[i] * phi_hat[i, j]
            E.k2[i, j] = -1.0im * meshX.k2[j] * phi_hat[i, j]
        end
    end

    E.x1 .= real(ifft(E.k1, (2, 1)))
    E.x2 .= real(ifft(E.k2, (2, 1)))
 
end


function advectX(f :: Array{Complex{Float64}, 4},
                  E :: ElectricField2D,
                  meshX :: UniformMesh2D,
                  meshV :: UniformMesh2D,
                  delta_t :: Float64,
                  rho :: Array{Complex{Float64}, 2},
                  phi_hat :: Array{Complex{Float64}, 2})
    #update f
    fft!(f, (1, 2))
    @threads for i = 1:meshX.Nx1
        for j = 1:meshX.Nx2
            for k = 1:meshV.Nx1
                for l = 1:meshV.Nx2
                    #update f
                    k_dot_v = meshV.x1[k] * meshX.k1[i] + meshV.x2[l] * meshX.k2[j]
                    f[i, j, k, l] = f[i, j, k, l] * exp(-1.0im * (k_dot_v) * delta_t)
                end
            end
        end
    end
    ifft!(f, (2, 1))
    f .= real(f)
    computeRho(rho, meshX, meshV, f)
    computePhiHat(phi_hat, meshX, rho)
    computeE(E, meshX, phi_hat)
end


function advect1DKernel( f,
                         alpha,
                         meshV,
                         delta_t)
    Nv = length(f)
    #println("Nv : $(Nv)")
    eval_points = zeros(Float64, Nv)
    f_old = copy(f)
    if (Nv == meshV.Nx1)
        f_old_interp = Spline1D(meshV.x1, real(f), k=3, bc="extrapolate", periodic=true)
        eval_points .= meshV.x1 .- (delta_t * alpha)
    else
    
        eval_points .= meshV.x2 .- (delta_t * alpha)
    end

    f .= f_old_interp(eval_points)
    #println("norminternal(f - f_old): $(norm(f - f_old))")
    #println("fnew: $(f)")
    #println("fold: $(f_old)")
    
    
end


function advectSL2DKernel(f, 
                          E1 :: Float64,
                          E2 :: Float64, 
                          meshV :: UniformMesh2D,
                          delta_t :: Float64)
    f_old_recons = Spline2D(meshV.x1, meshV.x2, real(f), kx=3, ky=3)
    # char_x1 = meshV.x1 .- (delta_t*E1)
    # char_x2 = meshV.x2 .- (delta_t*E2)
    for i = 1:meshV.Nx1, j = 1:meshV.Nx2
        f[i, j] = f_old_recons(meshV.x1[i] - (delta_t*E1), meshV.x2[j] - (delta_t*E2))
    end
    #println("Here!")
    
    #f .= evaluate(f_old_recons, char_x1, char_x2)
end



function advectVSL2(f :: Array{Complex{Float64}, 4},
                    E :: ElectricField2D,
                    meshX :: UniformMesh2D,
                    meshV :: UniformMesh2D,
                    delta_t :: Float64)
    #f_old = copy(real(f))
    #println("norm check: $(norm(real(f - f_old)))")
    f_slices = Array{Any}(undef, meshX.Nx1)
    @threads for i = 1:meshX.Nx1
        for j = 1:meshX.Nx2
            f_slices[i] = @views f[i, j, :, :]
            advectSL2DKernel(f_slices[i], E.x1[i, j], E.x2[i, j], meshV, delta_t)
        end
    end
    #println("norm change: $(norm(real(f - f_old)))")
    
end

function advectVSL(f :: Array{Complex{Float64}, 4},
                   E :: ElectricField2D,
                   meshX :: UniformMesh2D,
                   meshV :: UniformMesh2D,
                   delta_t :: Float64)

    #Lagrange splitting
    for i = 1:meshX.Nx1
        for j = 1:meshX.Nx2
            for k = meshV.Nx2
                f_old_recons = Spline1D(meshV.x1, real(f[i, j, :, k]), k = 3)
                f[i, j, :, k] .= f_old_recons(meshV.x1 .- E.x1[i, j] * delta_t)
            end
        end
    end
    
    for i = 1:meshX.Nx1
        for j = 1:meshX.Nx2
            for k = meshV.Nx1
                f_old_recons = Spline1D(meshV.x2, real(f[i, j, k, :]), k = 3)
                f[i, j, k, :] .= f_old_recons(meshV.x2 .- E.x2[i, j] * delta_t)
            end
        end
    end
    
    

    #println("norm change: $(norm(real(f - f_old)))")
        
end


function advectV(f :: Array{Complex{Float64}, 4},
                 E :: ElectricField2D,
                 meshX :: UniformMesh2D,
                 meshV :: UniformMesh2D,
                 delta_t :: Float64)
    #f_old = copy(real(f))
    fft!(f, (3, 4))
    @threads for i = 1:meshX.Nx1
        for j = 1:meshX.Nx2
            for k = 1:meshV.Nx1
                for l = 1:meshV.Nx2
                    #update f
                    E_dot_v = E.x1[i, j] * meshV.k1[k] + E.x2[i, j] * meshV.k2[l]
                    f[i, j, k, l] = f[i, j, k, l] * exp(-1.0im * (E_dot_v) * delta_t)
                end
            end
        end
    end
    ifft!(f, (4, 3))
    f .= real(f)
    #println("norm change: $(norm(real(f - f_old)))")
end

function computeKineticEnergy(f :: Array{Complex{Float64}, 4},
                              meshX :: UniformMesh2D,
                              meshV :: UniformMesh2D)
    dX = meshX.delta_x1 * meshX.delta_x2
    dV = meshV.delta_x1 * meshV.delta_x2
    
    intfdX = (sum(real(f), dims=(1, 2)) .* dX)[1, 1, :, :]
    energyKin = 0.0
    @threads for i = 1:meshV.Nx1
        for j=1:meshV.Nx2
            intfdX[i, j] = ((meshV.x1[i]^2 + meshV.x2[j]^2) * intfdX[i, j] * dV)
        end
    end
    energyKin = sum(intfdX)
    return energyKin
end

function computeElectricEnergy(E :: ElectricField2D,
                               meshX :: UniformMesh2D)
    dX = meshX.delta_x1 * meshX.delta_x2
    energyE = sum(((E.x1).^2 + (E.x2).^2) .* dX)
    return energyE
end


function vlasovAmpereSolveFourier(f :: Array{Complex{Float64}, 4},
                             E :: ElectricField2D,
                             meshX :: UniformMesh2D,
                             meshV :: UniformMesh2D,
                             delta_t :: Float64,
                             t_steps :: Int64)

    EnergyE = zeros(Float64, t_steps + 1)
    EnergyTotal = zeros(Float64, t_steps + 1)
    
    
    rho = zeros(Complex{Float64}, meshX.Nx1, meshX.Nx2)
    phi_hat = zeros(Complex{Float64}, meshX.Nx1, meshX.Nx2)

    EnergyE[1] = computeElectricEnergy(E, meshX)
    EnergyTotal[1] = EnergyE[1] + computeKineticEnergy(f, meshX, meshV)
    
    for ts in 1:t_steps
        
        advectV(f, E, meshX, meshV, 0.5*delta_t)
        
        advectX(f, E, meshX, meshV, delta_t, rho, phi_hat)

        EnergyE[ts+1] = computeElectricEnergy(E, meshX)
        EnergyTotal[ts+1] = EnergyE[ts+1] + computeKineticEnergy(f, meshX, meshV)
        println("Time step ts=$(ts), ts+1 =$(ts+1)")
                        
        advectV(f, E, meshX, meshV, 0.5*delta_t)
               
    end
    return EnergyE, EnergyTotal
    
end

function vlasovAmpereSolveSL(f :: Array{Complex{Float64}, 4},
                             E :: ElectricField2D,
                             meshX :: UniformMesh2D,
                             meshV :: UniformMesh2D,
                             delta_t :: Float64,
                             t_steps :: Int64)

    EnergyE = zeros(Float64, t_steps + 1)
    EnergyTotal = zeros(Float64, t_steps + 1)
    
    
    rho = zeros(Complex{Float64}, meshX.Nx1, meshX.Nx2)
    phi_hat = zeros(Complex{Float64}, meshX.Nx1, meshX.Nx2)

    EnergyE[1] = computeElectricEnergy(E, meshX)
    EnergyTotal[1] = EnergyE[1] + computeKineticEnergy(f, meshX, meshV)
    
    for ts in 1:t_steps
        
        advectVSL2(f, E, meshX, meshV, 0.5*delta_t)
        
        advectX(f, E, meshX, meshV, delta_t, rho, phi_hat)

        EnergyE[ts+1] = computeElectricEnergy(E, meshX)
        EnergyTotal[ts+1] = EnergyE[ts+1] + computeKineticEnergy(f, meshX, meshV)
        println("Time step ts=$(ts), ts+1 =$(ts+1)")
                        
        advectVSL2(f, E, meshX, meshV, 0.5*delta_t)
               
    end
    return EnergyE, EnergyTotal
    
end




