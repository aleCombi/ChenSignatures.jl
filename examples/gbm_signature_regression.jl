# Geometric Brownian Motion Signature Regression Example
#
# Demonstrates:
# 1. Simulating N paths of geometric Brownian motion
# 2. Time augmentation and batch signature computation
# 3. Regressing against call option payoff (S_T - K)^+
# 4. Visualization of regression results

# Disable plot display windows - only save to file (must be set before loading Plots)
ENV["GKSwstype"] = "nul"  # Non-interactive GR backend for Windows

using Revise
using ChenSignatures
using GLM
using GLMNet
using Plots
using Random
using Statistics
using LinearAlgebra

gr()  # Use GR backend

# ============================================================================
# Geometric Brownian Motion Simulation
# ============================================================================

# Simulate a single GBM path: dS = μS dt + σS dW
function simulate_gbm_path(S₀::Real, μ::Real, σ::Real, T::Real, N::Int;
                           rng=Random.GLOBAL_RNG)
    dt = T / (N - 1)
    path = Vector{Float64}(undef, N)
    path[1] = S₀

    for i in 2:N
        dW = sqrt(dt) * randn(rng)
        path[i] = path[i-1] * exp((μ - 0.5 * σ^2) * dt + σ * dW)
    end

    return path
end

# Simulate multiple GBM paths in batch format (N × 1 × num_paths)
function simulate_gbm_paths(S₀::Real, μ::Real, σ::Real, T::Real, N::Int,
                           num_paths::Int; rng=Random.GLOBAL_RNG)
    paths = Array{Float64, 3}(undef, N, 1, num_paths)

    for i in 1:num_paths
        paths[:, 1, i] = simulate_gbm_path(S₀, μ, σ, T, N; rng=rng)
    end

    return paths
end

# Time-augment paths: add time as explicit coordinate (1D → 2D)
# This enriches the signature feature space significantly
function time_augment_paths(paths::Array{Float64, 3}, T::Real)
    N, d, num_paths = size(paths)
    @assert d == 1 "Expected 1D paths, got dimension $d"

    augmented = Array{Float64, 3}(undef, N, 2, num_paths)
    time_grid = range(0, T, length=N)

    for i in 1:num_paths
        augmented[:, 1, i] = time_grid      # Time coordinate
        augmented[:, 2, i] = paths[:, 1, i] # Stock price
    end

    return augmented
end

# ============================================================================
# Payoff Functions
# ============================================================================

# Call option payoff: (S_T - K)^+
call_payoff(S_T::Real, K::Real) = max(S_T - K, 0.0)

# Extract terminal stock prices from batch of paths
function extract_terminal_values(paths::Array{Float64, 3})
    N, _, num_paths = size(paths)
    return [paths[N, 1, i] for i in 1:num_paths]
end

# Compute call option payoffs for all paths
function compute_payoffs(paths::Array{Float64, 3}, K::Real)
    terminal_values = extract_terminal_values(paths)
    return call_payoff.(terminal_values, K)
end

# ============================================================================
# Signature Computation
# ============================================================================

# Compute signatures for batch of paths using efficient batch processing
function compute_signatures_batch(paths::Array{Float64, 3}, level::Int)
    return sig(paths, level; threaded=true)
end

# ============================================================================
# Regression
# ============================================================================

# Fit regression model: signatures → payoffs
# Supports OLS, Ridge (alpha=0), Lasso (alpha=1), and Elastic Net (0 < alpha < 1)
function fit_signature_regression(signatures::Matrix{Float64}, targets::Vector{Float64};
                                  regression_type::Symbol=:ols)
    X = signatures'  # Convert to (num_samples, num_features)

    if regression_type == :ols
        # Ordinary Least Squares via GLM
        return lm(X, targets)
    elseif regression_type in [:ridge, :lasso, :elastic_net]
        # Ridge (alpha=0), Lasso (alpha=1), or Elastic Net via GLMNet
        # Use cross-validation to select optimal lambda
        alpha = regression_type == :ridge ? 0.0 : (regression_type == :lasso ? 1.0 : 0.5)
        cv_result = glmnetcv(X, targets, alpha=alpha)
        return (model=cv_result, type=regression_type, lambda=cv_result.lambda[argmin(cv_result.meanloss)])
    else
        error("Unknown regression_type: $regression_type. Use :ols, :ridge, :lasso, or :elastic_net")
    end
end

# Predict using fitted model
function predict_regression(model, signatures::Matrix{Float64})
    X = signatures'

    if model isa GLM.LinearModel || model isa StatsModels.TableRegressionModel
        # OLS from GLM
        return GLM.predict(model, X)
    else  # GLMNet model (named tuple with model field)
        # Use lambda with minimum CV error
        best_idx = argmin(model.model.meanloss)
        return vec(GLMNet.predict(model.model.path, X, best_idx))
    end
end

# Compute R² score
function compute_r_squared(y_true::Vector{Float64}, y_pred::Vector{Float64})
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    return 1 - ss_res / ss_tot
end

# ============================================================================
# Visualization
# ============================================================================

# Plot predicted vs actual payoffs as function of terminal stock price
function plot_regression_results(S_T::Vector{Float64},
                                payoffs_true::Vector{Float64},
                                payoffs_pred::Vector{Float64},
                                K::Real,
                                sig_level::Int)
    # Sort by terminal price for cleaner visualization
    perm = sortperm(S_T)
    S_T_sorted = S_T[perm]
    true_sorted = payoffs_true[perm]
    pred_sorted = payoffs_pred[perm]

    # Create plot
    plt = scatter(S_T_sorted, true_sorted,
                  label="True Payoff",
                  alpha=0.5,
                  xlabel="Terminal Stock Price (S_T)",
                  ylabel="Payoff",
                  title="Signature Regression: Predicted vs True Payoffs",
                  legend=:topleft,
                  markersize=2)

    scatter!(plt, S_T_sorted, pred_sorted,
             label="Predicted (sig level=$sig_level)",
             alpha=0.5,
             markersize=2)

    # Add strike price line
    vline!(plt, [K],
           label="Strike (K=$K)",
           linestyle=:dash,
           linewidth=2,
           color=:black)

    return plt
end

# ============================================================================
# Main Example
# ============================================================================

function run_gbm_signature_regression_example(;
    S₀=100.0,
    μ=0.05,
    σ=0.2,
    T=1.0,
    K=100.0,
    N=50,
    train_paths=400000,
    test_paths=200,
    plot_paths=500,  # Additional paths for plotting
    sig_level=7,
    seed=42,
    samples_per_feature=20,  # Rule of thumb: 10-20 samples per feature
    regression_type::Symbol=:ols  # :ols, :ridge, :lasso, :elastic_net
)
    println("=" ^ 70)
    println("Geometric Brownian Motion Signature Regression")
    println("=" ^ 70)
    println()

    # Calculate signature dimension: d + d² + ... + d^m for d=2 (time-augmented)
    sig_dim = sum(2^k for k in 1:sig_level)
    min_samples = sig_dim * samples_per_feature

    # Adjust training samples if needed to avoid overfitting
    if train_paths < min_samples
        train_paths = min_samples
        println("⚠ Signature dimension ($sig_dim) requires at least $min_samples samples")
        println("  (using $samples_per_feature samples per feature rule)")
        println("  Automatically increasing train_paths to $train_paths")
        println()
    end

    rng = MersenneTwister(seed)

    # ========================================================================
    # 1. Simulate training paths
    # ========================================================================
    println("1. Simulating $train_paths training paths...")
    train_paths_data = simulate_gbm_paths(S₀, μ, σ, T, N, train_paths; rng=rng)
    println("   Shape: $(size(train_paths_data))")
    println()

    # ========================================================================
    # 2. Time-augment and compute signatures
    # ========================================================================
    println("2. Time-augmenting paths and computing signatures (level $sig_level)...")
    train_paths_augmented = time_augment_paths(train_paths_data, T)
    train_signatures = compute_signatures_batch(train_paths_augmented, sig_level)
    println("   Signature dimension: $sig_dim")
    println("   Samples/feature ratio: $(round(train_paths/sig_dim, digits=1))")
    println()

    # ========================================================================
    # 3. Compute payoffs and fit regression
    # ========================================================================
    println("3. Fitting $regression_type regression model (signatures → payoffs)...")
    train_payoffs = compute_payoffs(train_paths_data, K)
    model = fit_signature_regression(train_signatures, train_payoffs; regression_type=regression_type)

    # Print lambda if using regularized regression
    if regression_type != :ols
        println("   Selected λ: $(round(model.lambda, sigdigits=3)) (via CV)")
    end

    train_predictions = predict_regression(model, train_signatures)
    train_r2 = compute_r_squared(train_payoffs, train_predictions)
    println("   Training R²: $(round(train_r2, digits=4))")
    println()

    # ========================================================================
    # 4. Test on separate test set
    # ========================================================================
    println("4. Testing on $test_paths new paths...")
    test_paths_data = simulate_gbm_paths(S₀, μ, σ, T, N, test_paths; rng=rng)
    test_paths_augmented = time_augment_paths(test_paths_data, T)
    test_signatures = compute_signatures_batch(test_paths_augmented, sig_level)
    test_payoffs = compute_payoffs(test_paths_data, K)
    test_predictions = predict_regression(model, test_signatures)
    test_r2 = compute_r_squared(test_payoffs, test_predictions)
    println("   Test R²: $(round(test_r2, digits=4))")
    println("   MAE: $(round(mean(abs.(test_payoffs .- test_predictions)), digits=2))")
    println()

    # ========================================================================
    # 5. Generate visualization on separate plot set
    # ========================================================================
    println("5. Generating visualization on $plot_paths new paths...")
    plot_paths_data = simulate_gbm_paths(S₀, μ, σ, T, N, plot_paths; rng=rng)
    plot_paths_augmented = time_augment_paths(plot_paths_data, T)
    plot_signatures = compute_signatures_batch(plot_paths_augmented, sig_level)
    plot_payoffs = compute_payoffs(plot_paths_data, K)
    plot_predictions = predict_regression(model, plot_signatures)
    plot_S_T = extract_terminal_values(plot_paths_data)

    plt = plot_regression_results(plot_S_T, plot_payoffs, plot_predictions, K, sig_level)
    savefig(plt, "gbm_signature_regression.png")
    println("   Plot saved as 'gbm_signature_regression.png'")
    println()

    # ========================================================================
    # Summary
    # ========================================================================
    println("=" ^ 70)
    println("Summary")
    println("=" ^ 70)
    println("GBM Parameters: S₀=$S₀, μ=$μ, σ=$σ, T=$T")
    println("Option: Call with strike K=$K")
    println("Paths: $train_paths training, $test_paths test, $plot_paths plot")
    println("Time augmentation: 1D → 2D (time, price)")
    println("Signature level: $sig_level (dimension: $sig_dim)")
    println("Regression: $regression_type" * (regression_type != :ols ? " (λ=$(round(model.lambda, sigdigits=3)))" : ""))
    println()
    println("Results:")
    println("  Training R²: $(round(train_r2, digits=4))")
    println("  Test R²:     $(round(test_r2, digits=4))")
    println("=" ^ 70)

    return (model=model, train_r2=train_r2, test_r2=test_r2, plot=plt)
end

# ============================================================================
# Run the example
# ============================================================================

# Run automatically when script is executed or included
run_gbm_signature_regression_example(
    S₀=100.0,
    μ=0.05,
    σ=0.2,
    T=1.0,
    K=100.0,
    N=50,
    train_paths=100000,
    test_paths=200,
    plot_paths=500,
    sig_level=10,
    seed=42,
    samples_per_feature=20,
    regression_type=:ridge
)
