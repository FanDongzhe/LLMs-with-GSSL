
for feature_type in 'ogb' 'TA' 'E' 'P'; doo
    count=0
    for dataset in 'cora' 'pubmed' 'arxiv'; do #'cora' 'pubmed' 'arxiv'; d
        python -u execute.py --dataset $dataset --aug_type subgraph --drop_percent 0.20  \
        --save_name ${dataset}_${feature_type}_best_dgi.pkl --gpu $count --feature_type $feature_type \
        2>&1 | tee out/${dataset}_${feature_type}.out &
        count=$((count+1))
    done
    wait
done
wait