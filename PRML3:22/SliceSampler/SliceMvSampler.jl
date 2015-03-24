using PyPlot, Distributions

find_interval(x0::Float64 ,w::Float64 ,u::Float64, g::Function) = begin

    Z_min::Float64 = x0 - w*rand()
    Z_max::Float64 = x0 + w

    while u < g(Z_min)
        Z_min -= w
    end

    while u < g(Z_max)
      Z_max += w
    end
    return Z_min,Z_max

end

function sample_one(x0::Float64, g::Function, w::Float64, gx0::Float64)

    u::Float64 = rand(Uniform(0,gx0))

    Z_min::Float64, Z_max::Float64 = find_interval(x0, w, u, g)

    x1::Float64 = 0.0
    gx1::Float64 = 0.0
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


#ここから違う
function mv_sample_one(x::Array{Float64}, f::Function,w=0.5)
  I = length(x)
  gxi = f(x)
  for i = 1:I
    #1変数関数
    function fi(w)
      x[i] = w
      f(x)
    end
    x[i], gxi = sample_one(x[i], fi, w, gxi)
  end
  x, gxi
end

function mv_slice_sampler(x::Array{Float64}, sampler::Function, niter::Int64)
  P = length(x)
  xs = zeros((niter,P))
  progress = zeros(niter)
  for iter = 1:niter
    x, fx = sampler(x, f)
    xs[iter,:] = x
    progress[iter] = fx
  end
  return xs, progress
end


function f(v::Array{Float64,2})
  x = v[1]
  y = v[2]
  if x < 0
    return -Inf
  end
  x^2 * exp(- x*y^2 - y^2 + 2*y - 4*x)
end

x0 = [0.1 0.5]

@time xs, progress = mv_slice_sampler(x0, mv_sample_one,  500);

using PyPlot

X = linspace(0, 4.)
Y = linspace(-1., 4.)
Z = Float64[f([x y])[1] for y = Y,x=X]
pcolor(X,Y,Z,alpha=0.5)

scatter(xs[:,1], xs[:,2])

