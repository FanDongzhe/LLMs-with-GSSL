for dataset in  'cora' 'pubmed' 'ogbn-arxiv'  ; do #'ogbn-arxiv' 'cora' 'pubmed'
    count=0
    for feature_type  in 'ogb' 'TA' 'E' 'P' ; do #  'ogb' 'TA' 'E' 'P'
        export CUDA_VISIBLE_DEVICES=${count} 
        python -u train_transductive.py --flagfile=config/${dataset}.cfg  --feature_type $feature_type 2>&1 | tee lr_0_01/${dataset}_${feature_type}.out &
        count=$((count+1))
    done
    wait
done
wait