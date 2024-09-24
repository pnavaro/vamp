include("vlasov-ampere2D.jl")

function landau1D( epsilon, kx, x, v )
    
    (1.0.+ epsilon*cos.(kx*x))/sqrt(2*pi) .* transpose(exp.(-0.5*v.*v))
    
end


function test_initLandau2D(f :: Array{Complex{Float64}, 4},
                            epsilon :: Float64,
                            k_x1 :: Float64,
                            k_x2 :: Float64,
                            meshX :: UniformMesh2D,
                            meshV :: UniformMesh2D)
    l1D = landau1D(epsilon, k_x1, meshX.x1, meshV.x1)
    test_mesh = UniformMesh2D(meshV.x1_min, meshV.x1_max,
                              meshV.x2_min, meshV.x2_max,
                              meshV.Nx1, meshV.Nx2)
    test_mesh.x2 = zeros(Float64, test_mesh.Nx2)
    initLandau2D(f, epsilon,
                  k_x1, 0.0,
                  meshX,
                  test_mesh)
    l1D = l1D./sqrt(2*pi)

    println("Testing initialization of landau damping 2D")
    println("$(norm(l1D - f[:, 1, :, 1]))")
end



