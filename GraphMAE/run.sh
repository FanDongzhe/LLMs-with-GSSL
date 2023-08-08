for dataset in  'pubmed'  ; do #'ogbn-arxiv' 'cora' 'pubmed'
    count=1
    for feature_type  in 'ogb' ; do #  'ogb' 'TA' 'E' 'P'
        export CUDA_VISIBLE_DEVICES=${count} 
        sh scripts/run_transductive.sh $dataset 0 $feature_type 2>&1 | tee raw/${dataset}_${feature_type}.out &
        count=$((count+1))
    done
done
wait