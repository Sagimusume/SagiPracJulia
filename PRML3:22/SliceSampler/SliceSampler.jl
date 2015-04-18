using PyPlot, Distributions

#Lower Bound
S(logy,g) = x-> logy < g(x)

#Find a Interval
find_interval(x0::Float64 ,w::Float64 ,is_in_S::Function) = begin

    L::Float64 = x0 - w*rand()
    R::Float64 = x0 + w

    while is_in_S(L)
        L -= w
    end

    while is_in_S(R)
      R += w
    end
    return L,R

end

#SliceSampler(OneStep)
function slice_sampler(x0::Float64, g::Function, w::Float64, gx0::Float64)

    logy::Float64 = gx0 - rand(Exponential(1.0))

    is_in_S::Function = S(logy,g)

    L::Float64, R::Float64 = find_interval(x0, w, is_in_S)

    x1::Float64 = 0.0
    gx1::Float64 = 0.0
    while true
        x1 = rand() * (R-L) + L#Sample form a Uniform (L<=I<=R)
        gx1 = g(x1)::Float64

        if gx1 >= logy
          break
        end

        if x1 > x0
          R = x1
        else
          L = x1
        end
    end

    return x1,gx1
end

#Sampler
function mcmc(x,g,N)
    xs = zeros(N)
    gx = g(x)
    for i = 1:N
        x,gx = slice_sampler(x ,g ,0.5 ,gx)
        xs[i] = x
    end
    return xs
end

#log Mixtur of Gaussian
f(x) = log(0.75 * pdf(Normal(0.0, 1.0), x) + 0.25 * pdf(Normal(8.0, 1.0), x))

#Main
m = mcmc(0.0, f, 100000)

#Plot
n, bins, patches = PyPlot.hist(m,bins=100,normed=true)
line = exp(f(bins))
PyPlot.plot(bins,line,"r--",linewidth=1)
