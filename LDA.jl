using Bigram

type Lda
	iteration_number::Int64
	topic_number::Int64
	alpha::Float64
	beta::Float64
	index::Dict{String,Int64}
	index_number::Int64
	document_number::Int64
	doc::Array{String,1}
	n_dz::Array{Float64,2}
	n_tz::Array{Float64,2}
	n_z::Array{Float64,1}
	term_topic::Array{Array{Int64,1},1}
    tmp_prob::Array{Float64,1}
	function Lda(iteration_number,topic_number,alpha,beta)

			ind_dir,doc = read_document()
			unique_index = map(string,unique(join(doc)))

			document_number = length(ind_dir)
			index_number = length(unique_index)
			index = Dict{String,Int}(zip(unique_index,1:index_number))

			n_dz = zeros(document_number,topic_number) .+ alpha
			n_tz = zeros(topic_number,index_number).+ beta
			n_z  = zeros(topic_number)
			term_topic = Array(Array{Int64,1},document_number)

            tmp_prob = Array(Float64,topic_number)

			new(iteration_number,topic_number,alpha,beta,
				index,index_number,document_number,
				doc,n_dz,n_tz,n_z,term_topic,tmp_prob)
	end
end


function initialize(l::Lda)
	for (i,j) = enumerate(l.doc)
		l.term_topic[i] = rand(1:l.topic_number,length(l.doc[i]))
		for (k,m) = enumerate(l.doc[i])
			l.n_dz[i,l.term_topic[i][k]] += 1
			l.n_tz[l.term_topic[i][k],l.index[string(m)]] += 1
			l.n_z[l.term_topic[i][k]] += 1
		end
	end
end

function decrement(l::Lda,d_i::Int64,t_j::Int64,this_term_topic::Int64, this_term_index::Int64)
	l.n_dz[d_i,this_term_topic] -= 1
	l.n_tz[this_term_topic,this_term_index] -= 1
	l.n_z[this_term_topic] -= 1
end

function increment(l::Lda,d_i::Int64,t_j::Int64,new_term_topic::Int64,this_term_index::Int64)
	l.n_dz[d_i,new_term_topic] += 1
	l.n_tz[new_term_topic,this_term_index] += 1
	l.n_z[new_term_topic] += 1
end

function compute_new_topic(l::Lda,d_i::Int64, t_j::Int64, this_term_index::Int64)
	for topic_k = 1:l.topic_number
		l.tmp_prob[topic_k] = l.n_dz[d_i,topic_k] * (l.n_tz[topic_k,this_term_index]/l.n_z[topic_k])
	end
	new_topic = SampleOne(l.tmp_prob)
end

function SampleOne(probs)
    z = sum(probs)
    remaining = rand()*z
    for i = 1:length(probs)
        remaining -= probs[i]
        if remaining < 0
            return i
        end
    end
end


function iterator(v::Lda)
	for itr = 1:v.iteration_number
		for i = 1:length(v.doc)
			for (k,l) = enumerate(v.doc[i])
				this_term_index = v.index[string(l)]
				this_term_topic = v.term_topic[i][k]

				decrement(v,i,k,this_term_topic,this_term_index)
				new_term_topic = compute_new_topic(v,i,k,this_term_index)
				increment(v,i,k, new_term_topic, this_term_index)

				v.term_topic[i][k] = new_term_topic
			end
		end
        println(100*itr/v.iteration_number,"%")
	end
end

l = Lda(1500,100,0.5,0.5);
initialize(l);
@time iterator(l);
