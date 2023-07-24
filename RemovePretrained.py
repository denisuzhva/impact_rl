import yaml
import os
import sys



if len(sys.argv) == 1:
    configs = []
else:
    configs = sys.argv[1:]

cfg_path = f'./cfg/'

chkp_dir = f'./checkpoints/'
chkp_fnames = os.listdir(chkp_dir)
log_dir = f'./train_logs/'

input(f"Press Enter to remove trained models: {configs}")

for cfg_name in configs:
    try:
        for fname in chkp_fnames:
            if cfg_name in fname:
                os.remove(chkp_dir + fname)
        os.remove(log_dir + f'{cfg_name}_log.csv')
        print(f"Success! Experiment {cfg_name} removed")
    except OSError as error:
        print(error)
        print("File path cannot be removed")  
    