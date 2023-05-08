import yaml
import os



cfg_path = f'./prototype2_csnet/cfg/'
with open(cfg_path + 'general.yaml') as f:
    general_cfg = yaml.safe_load(f)

chkp_dir = f'./prototype2_csnet/checkpoints/'
chkp_fnames = os.listdir(chkp_dir)
log_dir = f'./prototype2_csnet/train_logs/'

runs = general_cfg['runs']
input(f"Press Enter to remove trained models: {runs}")

for run_name in runs:
    try:
        for fname in chkp_fnames:
            if run_name in fname:
                os.remove(chkp_dir + fname)
        os.remove(log_dir + f'{run_name}_log.csv')
        print(f"Success! Experiment {run_name} removed")
    except OSError as error:
        print(error)
        print("File path cannot be removed")  
    