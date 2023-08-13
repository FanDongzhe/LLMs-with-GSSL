for dataset in 'amazon-history' 'amazon-computers' 'amazon-photo'; do 
    count=0
    for feature_type in 'BOW' 'W2V'; do 
        export CUDA_VISIBLE_DEVICES=${count}  
        python -u s2gae_nc_acc.py --dataset $dataset --feature_type $feature_type --eval_multi_k 2>&1 | tee eval/${dataset}_${feature_type}.out &
        count=$((count+1))
    done
    wait
done



