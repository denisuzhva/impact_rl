import torch
from torch.utils.data import DataLoader

import os
from os.path import exists
import numpy as np
import pandas as pd
import yaml
from pprint import pprint

from q_trainer import train_q_agent
import environments.q_playground
import agents.q_agent 
from agents.q_agent import ExperienceReplay
import nets.q_nets



if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Best available device:', device)
    
    cfg_path = f'./cfg/'
    
    # Load general config
    with open(cfg_path + 'general.yaml') as f:
        general_cfg = yaml.safe_load(f)
    
    run_names = general_cfg['runs']
    for run_name in run_names:
        print(run_name)
        cfg_path_run = cfg_path + run_name + '/'
        with open(cfg_path_run + 'env.yaml') as f:
            env_data = yaml.safe_load(f)
        with open(cfg_path_run + 'agents.yaml') as f:
            agent_data = yaml.safe_load(f)
        with open(cfg_path_run + 'models.yaml') as f:
            all_models_data = yaml.safe_load(f)
        with open(cfg_path_run + 'trainer.yaml') as f:
            trainer_params = yaml.safe_load(f)
            
        # Create environment
        env_name = env_data['name']
        env_class = getattr(environments.q_playground, env_name)
        env = env_class(env_data['img_path'],
                        env_data['reward_weight'],
                        env_data['flatten']) 
        env_state_size = env.state_size
        
        # Define the agents and Q-networks
        agents_list = []
        n_agents = agent_data['n_agents']
        for adx in range(n_agents):
            agents_list.append({})
            buffer = ExperienceReplay(trainer_params['dql_params']['replay_start_size'])
            agent_class_name = agent_data['agent_class']
            agent_class = getattr(agents.q_agent, agent_class_name)
            agents_list[adx]['agent'] = agent_class(env, buffer)
            for model_type, model_data in all_models_data.items():
                model_class_name = model_data['model_class']
                model_class = getattr(nets.q_nets, model_class_name)
                model_data['params']['in_size'] = env_state_size
                model_data['params']['out_size'] = env_state_size
                model = model_class(model_data['params'])
                model.to(device)
                agents_list[adx][model_type] = model
 
        # Log and model checkpoints
        train_log_dir = './train_logs/'
        os.makedirs(train_log_dir, exist_ok=True)
        train_log_path = train_log_dir + run_name + '_log.csv'

        trained_dump_dir = './checkpoints/'
        os.makedirs(trained_dump_dir, exist_ok=True)
        opt_path = trained_dump_dir + run_name + '_opt.pth'
        
        # Check if model and optimizer dump exists
        if exists(opt_path):
            print("Models and opt loaded")
            opt_chkp = torch.load(opt_path)
            for adx in range(n_agents):
                #for model_type in all_models_data.keys():
                model.load_state_dict(torch.load(trained_dump_dir + f'{run_name}_{adx}_policy_net.pth'), strict=False)
        else:
            print("Models and opt not found! Initiating new training instance")
            opt_chkp = None
        
        # Check if log exists
        if os.path.exists(train_log_path):
            print("Log loaded")
            log_df = pd.read_csv(train_log_path)
            #last_frame_idx = log_df['frame_idx'].iloc[-1]
            #min_v_loss = log_df['min_v_loss'].iloc[-1]
        else:
            print("Initiating new log")
            log_df = None
            #last_frame_idx = 0
            #min_v_loss = np.Inf

        # Train the first agent
        if trainer_params['do_train']:
            train_q_agent(
                agents_list[0],
                learning_rate_params=trainer_params['lr_params'],
                dql_params=trainer_params['dql_params'],
                crit_lambdas=trainer_params['losses'],
                device=device,
                run_name=run_name,
                log_df=log_df,
                log_df_path=train_log_path,
                trained_dump_dir=trained_dump_dir,
                opt_path=opt_path,
                #last_frame_idx=last_frame_idx,
                #min_v_loss=min_v_loss,
                opt_chkp=opt_chkp,
            )
            
