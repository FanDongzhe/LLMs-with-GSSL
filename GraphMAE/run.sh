for dataset in 'ogbn-arxiv' 'cora' 'pubmed' ; do #'ogbn-arxiv''cora' 'pubmed'
    count=0
    for feature_type  in 'ogb' 'TA' 'E' 'P'; do
        export CUDA_VISIBLE_DEVICES=${count} 
        sh scripts/run_transductive.sh $dataset 0 $feature_type 2>&1 | tee out/${dataset}_${feature_type}.out &
        count=$((count+1))
    done
done
wait