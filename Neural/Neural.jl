module Neural

    export Model, ANN, QNN, LQNN, quasi_newton_method, sgd, l_bfgsNN,
            preprocess, predict, gradient, getWeights
    # Artificial neural network
    using NumericExtensions

    #クラス定義
    include("base.jl")
    #アクセサ・ゲッタ
    include("ann_weight.jl")
    #バック・フォワードプロパゲーション
    include("bfprop.jl")
    #確率的勾配降下法
    include("sgd_methods.jl")
    #ラインサーチ
    include("linesearch.jl")
    #ラインサーチユーティリティ
    include("linesearch_utils.jl")
    #準ニュートン法(定数定義あり)
    include("quasi_methods.jl")
    #L−BFGS
    include("l_BFGS.jl")
    #ユーティリティ
    include("utils.jl")

#ModuleEnd
end

