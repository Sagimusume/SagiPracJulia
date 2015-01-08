#Julia v0.3.3 Mac Yosemite 10.10.1
# MacBook Air Mid 2013 Intel Core i5
using Bigram
######################################################
# 1. Bigramを参照
# 2. Bigramと同じディレクトリ内に配置(カレントディレクトリetc.)
# 3. HMM(イテレーション回数,Tagの種類数)
# 4. julia Repl　内で　requir("HMM.jl")と入力
# 5. Wos_num が　得られたTag　のベクトル
######################################################
type HMM
	Iteration_num::Int64
	Wos_num::Int64 #品詞数
	Document::Array{Char,1}
	Index::Dict{String,Int64}
	S::Array{Float64,2}
	I::Array{Float64,2}
	Wos_Vec::Array{Int64,1}
	p::Array{Float64,1}
	s_1::Array{Float64,1}

	function HMM(Iteration_num,Wos_num)
		ind_dir,Document = read_document()
        Document = join(Document)|>collect
		Index = CreatIndex(Document)
		S = zeros(Wos_num,Wos_num).+0.5 #品詞*品詞
		I = zeros(length(Index),Wos_num).+0.5#品詞*Vocablary
		Wos_Vec = rand(1:Wos_num,length(Document))
		unshift!(Wos_Vec,1),push!(Wos_Vec,1)
		p = Array(Float64,Wos_num)
		new(Iteration_num,Wos_num,Document,Index,S,I,Wos_Vec,p)
	end

end


function Initialize(hmm::HMM)
	for (doc,j) = zip(hmm.Document,[2:length(hmm.Wos_Vec)-1])
		y_m1,y,y_p1 = hmm.Wos_Vec[j-1],hmm.Wos_Vec[j],hmm.Wos_Vec[j+1]
		hmm.S[y_m1,y] += 1
		hmm.I[hmm.Index[string(doc)],y] += 1
	end
    hmm.S[hmm.Wos_Vec[end-1],1] += 1
	hmm.s_1 = hmm.S*ones(hmm.Wos_num)
end


function Decrement(hmm::HMM,y_m1,y,y_p1,this_term_index)
		hmm.S[y_m1,y] -= 1
		hmm.S[y,y_p1] -= 1
		hmm.I[this_term_index,y] -= 1
		hmm.s_1[y] -= 1
end

function Increment(hmm::HMM,y_m1,new_y,y_p1,this_term_index)
		hmm.S[y_m1,new_y] += 1
		hmm.S[new_y,y_p1] += 1
		hmm.I[this_term_index,new_y] += 1
		hmm.s_1[new_y] += 1
end

function SampleOne(probs)
	z = sum(probs)
	remaining = rand()* z
	for i = 1:length(probs)
		remaining -= probs[i]
		if remaining < 0
			return i
		end
	end
end

function compute_new_p(hmm::HMM,y_m1,y,y_p1,this_term_index)
	for tag = 1:hmm.Wos_num
		hmm.p[tag] = (hmm.S[y_m1,tag]/hmm.s_1[y_m1])*
				 	 (hmm.S[tag,y_p1]/hmm.s_1[tag])*
					 (hmm.I[this_term_index,tag]/hmm.s_1[tag])
	end
	new_p = SampleOne(hmm.p)
end

function Iterator(hmm::HMM)
	for i = 1:hmm.Iteration_num
		for (doc,j) = zip(hmm.Document,[2:length(hmm.Wos_Vec)-1])
			y_m1,y,y_p1 = hmm.Wos_Vec[j-1],hmm.Wos_Vec[j],hmm.Wos_Vec[j+1]
			this_term_index = hmm.Index[string(doc)]
			Decrement(hmm,y_m1,y,y_p1,this_term_index)
			new_y = compute_new_p(hmm,y_m1,y,y_p1,this_term_index)
			Increment(hmm,y_m1,new_y,y_p1,this_term_index)
			hmm.Wos_Vec[j] = new_y
		end
        println(100*i/hmm.Iteration_num,"%")
	end
end
hmm = HMM(1500,20);
Initialize(hmm);
@time Iterator(hmm)

