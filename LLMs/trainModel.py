from EmbTrainer import LMTrainer
import argparse
import torch
import pandas as pd


def load_data(file_name):

    if file_name == 'computers':
        file_path = "./dataset/Computers.csv"

    elif file_name == 'history':
        file_path = "./dataset/History_Final_with_BoW_embeddings.pt"

    elif file_name == 'photo':
        file_path = "./dataset/Photo_Final_with_BoW_embeddings.pt"


    
    df = pd.read_csv(file_path)

    text = list(df["text"])
    labels = list(df["label"])
    return text, labels

    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', type=str, default='cora')

    args = parser.parse_args()
    
    filename = args.filename

    text, labels = load_data(filename)



    seed = 0
    trainer = LMTrainer(seed, text, labels, filename)
    trainer.train()
    trainer.eval_and_save()
