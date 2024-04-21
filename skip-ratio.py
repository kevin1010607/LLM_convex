import numpy as np

data_path = 'llm_eval/result/skip_layer.txt'

with open(data_path, 'r') as fh:
    skip_layer_list = [int(line) for line in fh.readlines()]

    print(f'Max skip layer: {np.max(skip_layer_list): d}')
    print(f'Average skip layer: {np.mean(skip_layer_list): .3f} ± {np.std(skip_layer_list): .3f}')
    print(f'Total layer: 32')
    print(f'Skip ratio: {np.mean(skip_layer_list) / 32 * 100: .3f}%')
