count=0

for dataset in  'amazon-history' ; do #'amazon-history' 'amazon-computers' 'amazon-photo'
    for feature_type  in 'BOW' 'W2V' ; do #  'ogb' 'TA' 'E' 'P'
        export CUDA_VISIBLE_DEVICES=$(($count % 4))
        sh scripts/run_transductive.sh $dataset 0 $feature_type 2>&1 | tee out/${dataset}_${feature_type}.out &
        count=$((count+1))
    done
done
wait