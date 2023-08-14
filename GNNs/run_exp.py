import itertools
import subprocess


def run_GNNs_script(dataset_list, model_type_list, feature_type_list, epoch_list, k_shot_list, lr_list, weight_decay_list,dim_hidden_list):
    # 生成所有可能的参数组合
    all_combinations = itertools.product(dataset_list, model_type_list, feature_type_list, epoch_list, k_shot_list,
                                         lr_list, weight_decay_list, dim_hidden_list)

    for combination in all_combinations:
        dataset, model_type, feature_type, epoch, k_shot,lr,weight_decay,dim_hidden = combination

        # 构建指令
        cmd = f"python GNNs.py --dataset {dataset} --model_type {model_type} --feature_type {feature_type} --epoch {epoch} --k_shot {k_shot} --lr {lr} --weight_decay {weight_decay} --dim_hidden {dim_hidden}"

        print(f"Executing: {cmd}")

        # 执行指令
        subprocess.run(cmd, shell=True)


# 参数列表
dataset_list = ["amazon-photo"]
dim_hidden_list = [64]
model_type_list = ["GAT"]
feature_type_list = ["BOW"]
epoch_list = [500]
lr_list = [0.01]
weight_decay_list = [0.0005]
k_shot_list = [60,70,80,90,100,120,150]


# 执行脚本
run_GNNs_script(dataset_list, model_type_list, feature_type_list, epoch_list, k_shot_list, lr_list, weight_decay_list, dim_hidden_list)
