module Bigram
######################################################
# 1. dir を任意のディレクトリに変更
# 2. replace にて 正規表現を行い文章の整形
# 3. CreatIndex は　文字ボキャブラリ作成
######################################################
export remove,read_document,CreatIndex,Rindex

function read_document()
    dir  = "/Users/yoshiko/Dropbox/ZMQ/ZMQ/龍樹"
    cd(dir)
    ind_dir = readdir()
    #mac 特有のDS_Storeの削除
    shift!(ind_dir)
    doc = Array(String,length(ind_dir))
    for (i,j) = enumerate(ind_dir)
        # 2.replece
        doc[i] = replace(readall(j),r"[^亜-黑]","")
    end
    return ind_dir,doc
end

function CreatIndex(doc::Array{Char,1})
    Index = Dict{String,Int64}()
    doc = unique(doc)##文字に分割（not単語
    for (i,j) = enumerate(doc)
        Index[string(j)] = i
    end
    return Index
end

function CreatIndex(doc::UTF8String)
    Index = Dict{String,Int64}()
    doc = unique(doc)##文字に分割（not単語
    for (i,j) = enumerate(doc)
        Index[string(j)] = i
    end
    return Index
end

reg = ["&",";","<","#","/",">","\t","\n","、","　", "、" ,"」","「","：","；","？","，","『","』","！","《","》","]","[","*",")","(","〈","〉","】","【","＊",":",".",r"\w","◎"," "]
append!(reg,["\0", "\x01", "\x10", "\b", "%", "\x02", "\v", "@", "�", "\x04", "\x03", "`", "\x18"])
append!(reg,["{", "=", "\"", " ", "-",","])
append!(reg,["。"])
function remove(text)
    for i = reg
        text = replace(text,i,"")
    end
    return text
end


function Rindex(index)
    rindex = Dict{Int64,String}()
    for (i,j) = index
        rindex[j] = i
    end
    return rindex
end

#Modulend
end

