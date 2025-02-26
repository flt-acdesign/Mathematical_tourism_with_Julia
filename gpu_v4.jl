using SparseArrays
using CUDA
using IterativeSolvers
using LinearAlgebra
using Random

# Set random seed for reproducibility
Random.seed!(123)

"""
Generate a random sparse matrix and vector, then solve Ax = b using GPU-accelerated iterative solvers
"""
function sparse_gpu_solver(n::Int, density::Float64, solver_type::Symbol=:cg)
    # For very large matrices, adjust density to prevent out of memory errors
    if n > 100_000
        # Calculate maximum safe density based on available memory
        # For n=1,000,000, density of 0.001 would create ~1 billion nonzeros!
        adjusted_density = min(density, 10.0 / n)
        println("Adjusting density from $density to $adjusted_density for large matrix")
        density = adjusted_density
    end
    
    # Generate random sparse matrix A
    println("Generating random sparse matrix of size $n×$n with density $density...")
    A = sprand(n, n, density)
    
    # Make A appropriate for the selected solver
    if solver_type == :cg
        # For CG, the matrix needs to be symmetric positive definite
        # But we need to avoid full matrix multiplication for very large matrices
        if n > 100_000
            # For very large matrices, just add a diagonal shift 
            # This isn't truly SPD but will be diagonally dominant with positive diagonal
            println("Using diagonal shift for large matrix (approx. SPD)")
            diag_shift = 10.0 * ones(n)
            A = A + A' + spdiagm(0 => diag_shift)
        else
            # For smaller matrices, we can do proper SPD construction
            println("Making matrix symmetric positive definite for CG solver")
            A = A + A' + 10*I
        end
    else
        # For general solvers, ensure A is diagonally dominant
        println("Making matrix diagonally dominant for general solver")
        # Compute row sums efficiently for large matrices
        row_sums = vec(sum(abs.(A), dims=2))
        A = A + spdiagm(0 => row_sums .+ 1.0)
    end
    
    # Generate random vector b
    println("Generating random vector b of size $n...")
    b = rand(n)
    
    # Check if CUDA is functional
    use_gpu = CUDA.functional()
    if !use_gpu
        println("CUDA not functional, using CPU only...")
    else
        println("CUDA is functional, using GPU acceleration...")
    end
    
    # Set solver parameters
    tol = 1e-6
    maxiter = min(n, 1000)
    
    # Choose and execute solver
    println("Solving system using $solver_type solver...")
    
    if solver_type == :cg
        # Conjugate Gradient method (for symmetric positive definite matrices)
        
        if use_gpu
            # Implement CG directly on GPU
            println("Using custom GPU-accelerated CG implementation...")
            
            # For very large matrices, we need to be careful with GPU memory
            if n > 500_000 && CUDA.available_memory() < 8 * n * 8  # Need ~8 vectors of size n
                println("Matrix too large for GPU memory, using CPU instead...")
                time_solve = @elapsed begin
                    x = cg(A, b, abstol=1e-6, maxiter=maxiter)
                end
            else
                # Transfer data to GPU
                A_gpu = CUDA.CUSPARSE.CuSparseMatrixCSR(A)
                b_gpu = CuVector(b)
                x_gpu = CuVector(zeros(n))
                
                # Initial residual
                r_gpu = b_gpu - A_gpu * x_gpu
                p_gpu = copy(r_gpu)
                
                # Initialize variables
                rsold_gpu = dot(r_gpu, r_gpu)
                
                # Time the solver
                time_solve = @elapsed begin
                    for iter in 1:maxiter
                        Ap_gpu = A_gpu * p_gpu
                        alpha = rsold_gpu / dot(p_gpu, Ap_gpu)
                        x_gpu = x_gpu + alpha * p_gpu
                        r_gpu = r_gpu - alpha * Ap_gpu
                        
                        # Check convergence
                        rsnew_gpu = dot(r_gpu, r_gpu)
                        if sqrt(rsnew_gpu) < tol
                            println("CG converged after $iter iterations")
                            break
                        end
                        
                        # Update p for next iteration
                        p_gpu = r_gpu + (rsnew_gpu / rsold_gpu) * p_gpu
                        rsold_gpu = rsnew_gpu
                        
                        # Print progress periodically
                        if iter % 100 == 0
                            println("Iteration $iter, residual: $(sqrt(rsnew_gpu))")
                        end
                    end
                end
                
                # Transfer solution back to CPU
                x = Array(x_gpu)
            end
            
        else
            # Use standard CPU solver
            println("Using CPU-based CG implementation...")
            time_solve = @elapsed begin
                x = cg(A, b, abstol=1e-6, maxiter=maxiter)
            end
        end
        
    elseif solver_type == :bicgstab
        # BiCGSTAB method (for general matrices)
        
        if use_gpu
            # Implement BiCGSTAB directly on GPU
            println("Using custom GPU-accelerated BiCGSTAB implementation...")
            
            # For very large matrices, we need to be careful with GPU memory
            if n > 500_000 && CUDA.available_memory() < 10 * n * 8  # Need ~10 vectors of size n
                println("Matrix too large for GPU memory, using CPU instead...")
                time_solve = @elapsed begin
                    x = bicgstabl(A, b, 2, abstol=1e-6, maxiter=maxiter)
                end
            else
                # Transfer data to GPU
                A_gpu = CUDA.CUSPARSE.CuSparseMatrixCSR(A)
                b_gpu = CuVector(b)
                x_gpu = CuVector(zeros(n))
                
                # Initial residual
                r0_gpu = b_gpu - A_gpu * x_gpu
                r_gpu = copy(r0_gpu)
                p_gpu = copy(r0_gpu)
                
                # Initialize variables
                rho_prev = 1.0
                alpha = 1.0
                omega = 1.0
                
                # Time the solver
                time_solve = @elapsed begin
                    for iter in 1:maxiter
                        rho = dot(r0_gpu, r_gpu)
                        
                        beta = (rho / rho_prev) * (alpha / omega)
                        p_gpu = r_gpu + beta * (p_gpu - omega * A_gpu * p_gpu)
                        
                        v_gpu = A_gpu * p_gpu
                        alpha = rho / dot(r0_gpu, v_gpu)
                        
                        h_gpu = x_gpu + alpha * p_gpu
                        s_gpu = r_gpu - alpha * v_gpu
                        
                        t_gpu = A_gpu * s_gpu
                        omega = dot(t_gpu, s_gpu) / dot(t_gpu, t_gpu)
                        
                        x_gpu = h_gpu + omega * s_gpu
                        r_gpu = s_gpu - omega * t_gpu
                        
                        # Check convergence
                        res_norm = norm(r_gpu)
                        if res_norm < tol
                            println("BiCGSTAB converged after $iter iterations")
                            break
                        end
                        
                        rho_prev = rho
                        
                        # Print progress periodically
                        if iter % 100 == 0
                            println("Iteration $iter, residual: $res_norm")
                        end
                    end
                end
                
                # Transfer solution back to CPU
                x = Array(x_gpu)
            end
            
        else
            # Use standard CPU solver
            println("Using CPU-based BiCGSTAB implementation...")
            time_solve = @elapsed begin
                x = bicgstabl(A, b, 2, abstol=1e-6, maxiter=maxiter)
            end
        end
        
    elseif solver_type == :gmres
        # GMRES method (for general matrices)
        
        if use_gpu && n <= 100_000  # Only use GPU for GMRES on moderately sized problems
            # For GMRES, switch to CPU implementation
            # Implementing a proper GPU GMRES without scalar indexing is complex
            println("GMRES on GPU requires scalar indexing; using CPU instead...")
            time_solve = @elapsed begin
                x = gmres(A, b, restart=min(30, n), abstol=tol, maxiter=maxiter)
            end
        else
            # Use standard CPU solver
            println("Using CPU-based GMRES implementation...")
            time_solve = @elapsed begin
                x = gmres(A, b, restart=min(30, n), abstol=tol, maxiter=maxiter)
            end
        end
        
    else
        error("Unknown solver type: $solver_type")
    end
    
    # Verify solution
    residual = b - A * x
    residual_norm = norm(residual) / norm(b)
    println("Solver completed in $time_solve seconds")
    println("Relative residual norm: $residual_norm")
    
    return x, residual_norm, time_solve
end

"""
Generate a random sparse matrix and vector, then solve Ax = b
using GPU-accelerated iterative solvers with performance profiling
"""
function profile_sparse_solvers(n::Int=10_000, density::Float64=0.001)
    # Run with different solvers
    println("\n=== Running Conjugate Gradient Solver ===")
    x_cg, res_cg, time_cg = sparse_gpu_solver(n, density, :cg)

    println("\n=== Running BiCGSTAB Solver ===")
    x_bicgstab, res_bicgstab, time_bicgstab = sparse_gpu_solver(n, density, :bicgstab)

    println("\n=== Running GMRES Solver ===")
    x_gmres, res_gmres, time_gmres = sparse_gpu_solver(n, density, :gmres)

    # Compare solver performance
    println("\n=== Performance Comparison ===")
    println("Solver    | Residual        | Time (s)")
    println("----------|-----------------|----------")
    println("CG        | $(res_cg)       | $(time_cg)")
    println("BiCGSTAB  | $(res_bicgstab) | $(time_bicgstab)")
    println("GMRES     | $(res_gmres)    | $(time_gmres)")
    
    return Dict(
        :cg => (x_cg, res_cg, time_cg),
        :bicgstab => (x_bicgstab, res_bicgstab, time_bicgstab),
        :gmres => (x_gmres, res_gmres, time_gmres)
    )
end

"""
Solve a very large sparse system by automatically selecting
the most appropriate solver based on matrix properties
"""
function solve_very_large_system(n::Int=1_000_000, density::Float64=1e-6)
    println("\n=== Solving Very Large System ===")
    println("System size: $n×$n with initial density $density")
    
    # For very large systems, we need to use a much lower density
    # A million by million matrix with density 1e-6 still has ~1 trillion entries!
    # Adjust to a reasonable density that won't cause memory issues
    adjusted_density = min(density, 10.0 / n^2)
    println("Adjusting density to $adjusted_density for manageable memory usage")
    
    # First check matrix properties to choose best solver
    println("Creating test matrix to determine properties...")
    
    # Create a smaller test matrix with same properties to determine best solver
    test_n = min(n, 5_000)
    test_density = adjusted_density * (n / test_n)^2  # Scale density to maintain similar sparsity pattern
    A_test = sprand(test_n, test_n, test_density)
    
    # Check if matrix is symmetric
    is_symmetric = issymmetric(A_test)
    
    # Choose solver based on matrix properties
    if is_symmetric
        println("Matrix appears to be symmetric - using CG solver")
        solver_type = :cg
    else
        println("Matrix appears to be non-symmetric - using BiCGSTAB solver")
        solver_type = :bicgstab  # Changed from GMRES to BiCGSTAB for better GPU compatibility
    end
    
    # Solve the actual large system
    x, res, time_solve = sparse_gpu_solver(n, adjusted_density, solver_type)
    
    return x, res, time_solve, solver_type
end

# Run performance benchmarks on a moderately large system
results = profile_sparse_solvers()

# Example of how to use for a very large system:
x_large, res_large, time_large, solver_large = solve_very_large_system(100_000, 1e-5)

# For an extremely large system (be cautious with memory):
x_large, res_large, time_large, solver_large = solve_very_large_system(1_000_000, 1e-7)