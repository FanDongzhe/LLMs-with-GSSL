for dataset in 'ogbn-arxiv' ; do #'ogbn-arxiv''cora' 'pubmed'
    for feature_type  in 'ogb' 'TA' 'E' 'P'; do
        export CUDA_VISIBLE_DEVICES=2 
        sh scripts/run_transductive.sh $dataset 0 $feature_type 2>&1 | tee out/${dataset}_${feature_type}.out 
    done
done
wait