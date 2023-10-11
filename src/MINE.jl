module MINE

using Zygote
using Zygote: @adjoint
using Distributions
using Flux
using Random
using Plots
using StatsFuns # for logsumexp
using LinearAlgebra
using HDF5
using Statistics

export compute_MI

function ema_loss_helper(t, running_mean)
    return StatsFuns.logsumexp(t .- log(size(t)[2]))
end

@adjoint ema_loss_helper(t, running_mean) = 
    ema_loss_helper(t, running_mean), G->(G*exp.(t.-log(size(t)[2]*(running_mean+1e-6))), nothing)

function ema_loss(t, running_mean, alpha)
    t_exp = exp( StatsFuns.logsumexp(t .- log(size(t)[2])) )
    
    if isapprox(running_mean, 0.0)
        running_mean = t_exp
    else
        running_mean = alpha * running_mean + (1-alpha) * t_exp
    end
    
    return ema_loss_helper(t, running_mean), running_mean
end

running_mean=0
function mutual_info(T, X, Y, Y_marg, alpha=0.9)
    global running_mean
    Z = vcat(X,Y)
    Z_prod = vcat(X,Y_marg)
    
    t = T(Z)
    t_prod = T(Z_prod)
    second_term, running_mean = ema_loss(t_prod, running_mean, alpha)
    
    return sum(t)/size(t)[2] - second_term
end

function compute_MI(X,Y; 
                    net_width=100, 
                    learning_rate=1e-4, 
                    num_steps=500, 
                    batch_size=128, 
                    test_fraction=0.1,
                    validation_fraction=0.1)
    
    # define model
    in_dim = size(X)[1] + size(Y)[1]
    T = Chain(
                Dense(in_dim => net_width, tanh),
                Dense(net_width => net_width, tanh),
                Dense(net_width => 1)
            )
    Flux.loadparams!(T, Flux.Params([Float64.(p) for p in Flux.params(T)]))
    
    optimal_params = copy(Flux.params(T))
    
    # organize data
    num_samples = size(X)[2]
    num_test = Int(floor(test_fraction * num_samples))
    num_valid = Int(floor(validation_fraction * num_samples))
    num_train = num_samples - num_test - num_valid
    
    X_train = X[:,1:num_train]
    Y_train = Y[:,1:num_train]
    X_test = X[:,num_train+1:num_train+num_test]
    Y_test = Y[:,num_train+1:num_train+num_test]
    X_valid = X[:,num_train+num_test+1:end]
    Y_valid = Y[:,num_train+num_test+1:end]
    
    loader = Flux.DataLoader((X_train, Y_train, Y_train[:,shuffle(1:end)]), batchsize=batch_size, shuffle=true);
    
    # training
    optim = Flux.setup(Flux.Adam(learning_rate), T)

    losses = []
    test_losses = []
    max_epoch = 1
    global running_mean = 0
    for epoch in 1:num_steps
        for (x,y,y_shuffled) in loader
            loss, grads = Flux.withgradient(T) do m
                # Evaluate model and loss inside gradient context:
                -mutual_info(m, x, y, y_shuffled)
            end
            Flux.update!(optim, T, grads[1])
            push!(losses, -loss)  # logging, outside gradient context
        end
        push!(test_losses, mutual_info(T, X_test, Y_test, Y_test[:,shuffle(1:end)]))
        if size(test_losses)[1]>1
            if max(test_losses[1:end-1]...) < test_losses[end]
                optimal_params = copy(Flux.params(T))
                max_epoch = epoch
            end
        end
    end
    
    if max_epoch == num_steps
        println("Warning: Optimization seems not converged.")
    end
    
    Flux.loadparams!(T, optimal_params)
    mi = [mutual_info(T, X_valid, Y_valid, Y_valid[:,shuffle(1:end)]) for _ in 1:25]

    return ( sum(mi)/25, std(mi), max(mi...), losses, optimal_params )
end

end # module MINE
