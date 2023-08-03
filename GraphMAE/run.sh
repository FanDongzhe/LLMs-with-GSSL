for dataset in 'cora' 'pubmed' 'ogbn-arxiv'; do
    for feature_type  in 'ogb' 'TA' 'E'; do
        sh scripts/run_transductive.sh $dataset 0 $feature_type 2>&1 | tee out/${dataset}_${feature_type}.out 
    done
done
wait