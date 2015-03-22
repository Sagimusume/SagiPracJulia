using PyPlot, Distributions
#z_min<= x_0 <= z_max
find_interval(x0::Float64 ,w::Float64 ,u::Float64, g::Function) = begin

    Z_min::Float64 = x0 - w/2
    Z_max::Float64 = x0 + w/2

    while u < g(Z_min)
        Z_min -= w
    end

    while u < g(Z_max)
      Z_max += w
    end
    return Z_min,Z_max

end

function sample_one(x0::Float64, g::Function, w::Float64, gx0::Float64)
    #Sample u
    u::Float64 = rand(Uniform(0,gx0))

    Z_min::Float64, Z_max::Float64 = find_interval(x0, w, u, g)
    #変数
    x1::Float64 = 0.0
    gx1::Float64 = 0.0
    #縮小ループ
    while true
        x1 = rand(Uniform(Z_min,Z_max))
        gx1 = g(x1)::Float64

        if gx1 >= u
          break
        end

        if x1 > x0
          Z_max = x1
        else
          Z_min = x1
        end
    end

    return x1,gx1
end

#Sampler
function slice_sampler(x,g,N)
    #x::InitialValue
    #g::Objective function
    #N::Sample Size
    xs = zeros(N)
    gx = g(x)
    #interval step size
    w = 0.5
    for i = 1:N
        x,gx = sample_one(x ,g ,w ,gx)
        xs[i] = x
    end
    return xs
end

#log Mixtur of Gaussian
f(x) = 0.75 * pdf(Normal(0.0, 1.0), x) + 0.25 * pdf(Normal(5.0, 1.0), x)

#Main
m = slice_sampler(-10.0, f, 5000)

#Plot
n, bins, patches = PyPlot.hist(m,bins=100,normed=true)
line = f(bins)
PyPlot.plot(bins,line,"r--",linewidth=1)
