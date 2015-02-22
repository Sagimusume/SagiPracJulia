function update(ann::Model, momentums::Vector)
    for l in 1:ann.L
        setWeights(ann, l, getWeights(ann, l) + momentums[l])
    end
end

function getDims(ann::Model)
    {(ann.dims[l+1], ann.dims[l]+1) for l in 1:ann.L}
end

function getWeights(ann::Model, layer::Integer)
    offset = 1
    for l in 1:layer-1
        offset += (ann.dims[l]+1) * ann.dims[l+1]
    end
    reshape(view(ann.w, offset:offset+(ann.dims[layer]+1)*ann.dims[layer+1]-1),
            ann.dims[layer+1], ann.dims[layer]+1)
end

function setWeights(ann::Model, layer::Integer, W::Matrix)
    offset = 1
    for l in 1:layer-1
        offset += (ann.dims[l]+1) * ann.dims[l+1]
    end
    ann.w[offset:offset+(ann.dims[layer]+1)*ann.dims[layer+1]-1] = W[:]
end