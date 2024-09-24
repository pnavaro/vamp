
# Vlasov-Ampere 2D-v, 2D-x prototype code
---

The test with Landau damping can be run with the following 
command: 

```
julia -i linear_landau_damping1.jl
```

This will create two pdf plots in the project directory:
1. Landau2DEE.pdf - a plot of electrical energy with time
2. Landau2DET.pdf - a plot of total energy error (absolute) with time.

The parameters for the landau damping simulation can be changed by editing the file: 
```
linear_landau_damping1.jl
```

In order to use threads, set the environment variable "JULIA_NUM_THREADS" in 
your shell. 
