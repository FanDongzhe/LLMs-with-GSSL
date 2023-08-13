export CUDA_VISIBLE_DEVICES=2

for feature_type in 'ogb'; do
    for dataset in 'ogbn-arxiv'; do #'cora' 'pubmed' 'arxiv'; d
        python -u execute.py --dataset $dataset --aug_type subgraph --drop_percent 0.20  \
        --save_name ${dataset}_${feature_type}_best_dgi.pkl --gpu 2 --feature_type $feature_type --eval_multi_k \
        2>&1 | tee out/${dataset}_${feature_type}.out 
    done
    wait
done
wait