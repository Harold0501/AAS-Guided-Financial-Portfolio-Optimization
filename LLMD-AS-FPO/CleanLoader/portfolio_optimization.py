import os
import argparse
import pickle

import torch

from src.utils import preprocess_data
from src.target_portfolio import OneHotWeightCollection, MaxSortinoWeightCollection


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', default = 'data/asset_1.csv', type = str)
    parser.add_argument('-n', '--num_epoch', default = 100, type = int)
    args = parser.parse_args()
    
    yearmons, data, torch_data, torch_target_data, missing_mask, num_assets = preprocess_data(args.file_name, device)

    file_base = os.path.splitext(os.path.basename(args.file_name))[0]

    weight_collection = OneHotWeightCollection(yearmons, num_assets, device)
    
    weight_collection.calculate_one_hot_weights(data, missing_mask)
    
    weights = weight_collection.get_weights()
    
    with open(f'result/{file_base}_weights_onehot.pkl', 'wb') as f:
        pickle.dump(weights, f)

    weight_collection = MaxSortinoWeightCollection(yearmons, num_assets, device)
    
    weight_collection.train_max_sortino_weights(data, missing_mask, num_epoch = args.num_epoch)
    
    weights = weight_collection.get_weights()
    
    with open(f'result/{file_base}_weights_maxsortino.pkl', 'wb') as f:
        pickle.dump(weights, f)
