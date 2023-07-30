# LLMs-with-GSSL

# SET UP
ADD folder prt_lm and dataset to ./BGRL

Can be download from https://drive.google.com/drive/folders/105qHQN4nOYkV8aGJqrHw5pyK8fpAUBCH?usp=drive_link

# To RUN BGRL
python train_transductive.py --flagfile=config/pubmed.cfg

# To RUN GraphCL
python execute.py --dataset pubmed --aug_type subgraph --drop_percent 0.20 --seed 1234 --save_name pubmed_best_dgi.pkl --gpu -1
