for dataset in 'Cora' 'Pubmed' 'arxiv'; do
    count=0
    for feature_type  in 'ogb' 'TA' 'E'; do
        ((count++))
        export CUDA_VISIBLE_DEVICES=${count}  
        python -u s2gae_nc_acc.py --dataset $dataset --feature_type $feature_type 2>&1 | tee ${dataset}_${feature_type}.out &
    done
done
wait