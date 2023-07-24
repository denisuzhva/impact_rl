import numpy as np
from src.FEM.solve_2 import solve_batch



if __name__ == '__main__':
    batch_size = 12
    config_list = []
    for idx in range(batch_size):
        config = np.zeros(24)
        config[0]=1
        config[1]=1
        config[2]=1
        config_list.append(config)
    v = solve_batch(config_list, data_dir='C:/dev/_spbu/impact_rl/FEM/temp/', ncpu_mp=6)
    print(v)