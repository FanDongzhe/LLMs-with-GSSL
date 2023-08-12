for dataset in  'ogbn-arxiv'  ; do #'ogbn-arxiv' 'cora' 'pubmed' 
    count=0
    for feature_type  in 'ogb' ; do #  'ogb' 'TA' 'E' 'P'
        export CUDA_VISIBLE_DEVICES=${count} 
        sh scripts/run_transductive.sh $dataset 0 $feature_type 2>&1 | tee eval/${dataset}_${feature_type}.out &
        count=$((count+1))
    done
    wait
done
wait