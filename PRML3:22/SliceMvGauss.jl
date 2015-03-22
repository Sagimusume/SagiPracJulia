using PyPlot, Distributions

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
function mv_sample_one(x::Array{Float64,1}, f::Function,w=0.5)
    D = length(x)#ベクトル次元
    gxi = f(x)   #初期パラメータ
    for i = 1:D
        #1変数関数化
        fi(d) = begin
            x[i] = d
            f(x)
        end
        x[i], gxi = sample_one(x[i], fi, w, gxi)
    end
    x, gxi
end

function mv_slice_sampler(x::Array{Float64,1}, sampler::Function, f::Function, N::Int64)
    D = length(x)
    xs = zeros(D,N)
    gxs = zeros(N)
    for iter = 1:N
        x, fx = sampler(x, f)
        xs[:,iter] = x
        gxs[iter] = fx
    end
    return xs, gxs
end

#目的関数定義
#MixMvGauss
#param1
ρ = -0.5
μ = [0., 0]
S  = [1. ρ; ρ 1]
Sⁱ = inv(S)
detS = det(S)
#param2
ρ2 = 0.9
μ2 = [2., -3.5]
S2  = [1. ρ2 ; ρ2 1]
Sⁱ2 = inv(S2)
detS2 = det(S2)

gx(x::Array{Float64,1}) = (1/3*exp(-(x-μ)'Sⁱ*(x-μ)/2)/(2π*sqrt(detS)) +
    2/3*exp(-(x-μ2)'Sⁱ2*(x-μ2)/2)/(2π*sqrt(detS2)))[1]

#Main
x0 = [-4., -8]

@time xs, progress = mv_slice_sampler(x0, mv_sample_one, gx, 1000);


#Plot
using PyPlot

X = linspace(-4, 6.,100)
Y = linspace(-8, 6.,100)
Z = Float64[gx([x, y])[1] for y = Y,x=X]
pcolor(X,Y,Z,alpha=0.5)
#plot_surface(X,Y,Z,alpha=0.5)
scatter(xs[1,:], xs[2,:])

