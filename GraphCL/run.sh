for dataset in 'cora'; do #'cora' 'pubmed' 'arxiv'; do
    count=0
    for feature_type in 'ogb' 'TA' 'E' 'P'; do
        python -u execute.py --dataset $dataset --aug_type subgraph --drop_percent 0.20 --seed 1234 \
        --save_name ${dataset}_best_dgi.pkl --gpu $count --feature_type $feature_type \
        2>&1 | tee out/${dataset}_${feature_type}.out
        count=$((count+1))
    done
    wait
done