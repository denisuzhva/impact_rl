import torch
#from torch.utils.data import DataLoader

import os
from os.path import exists
import sys
#import numpy as np
import pandas as pd
import yaml
#from pprint import pprint

from src.q_trainer import train_q_agent
import src.environments.q_playground as q_playground
import src.agents.q_agent as q_agent
import src.nets.q_nets as q_nets



if len(sys.argv) == 1:
    configs = ['FEM_3x6']
else:
    configs = sys.argv[1:]


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Best available device:', device)
    
    #cfg_path = 'C:/dev/_spbu/impact_rl/cfg/'
    cfg_path = './cfg/'
    
    for cfg_name in configs:
        print(cfg_name)
        cfg_path_run = cfg_path + cfg_name + '/'
        with open(cfg_path_run + 'env.yaml') as f:
            env_params = yaml.safe_load(f)
        with open(cfg_path_run + 'agents.yaml') as f:
            agent_params = yaml.safe_load(f)
        with open(cfg_path_run + 'models.yaml') as f:
            all_model_params = yaml.safe_load(f)
        with open(cfg_path_run + 'trainer.yaml') as f:
            trainer_params = yaml.safe_load(f)
            
        # Create environments and agents for each environment
        env_name = env_params['name']
        n_agents = agent_params['n_agents']
        env_class = getattr(q_playground, env_name)
        agent_data = {}
        agent_data['agent_list'] = []
        agent_class_name = agent_params['agent_class']
        for adx in range(n_agents):
            env = env_class(**env_params['kwargs'])
            agent_class = getattr(q_agent, agent_class_name)
            agent = agent_class(env)
            agent_data['agent_list'].append(agent)
        env_state_size = env.state_size
        
        # Define Q-nets
        for model_type, model_data in all_model_params.items():
            model_class_name = model_data['model_class']
            model_class = getattr(q_nets, model_class_name)
            model_data['params']['in_size'] = env_state_size
            model_data['params']['out_size'] = env_state_size
            model = model_class(model_data['params'])
            model.to(device)
            agent_data[model_type] = model
        
        # Define experience buffer
        buffer = q_agent.ExperienceReplay(trainer_params['dql_params']['replay_start_size'])
 
        # Log and model checkpoints
        #train_log_dir = 'C:/dev/_spbu/impact_rl/train_logs/'
        train_log_dir = './train_logs/'
        os.makedirs(train_log_dir, exist_ok=True)
        train_log_path = train_log_dir + cfg_name + '_log.csv'

        #trained_dump_dir = 'C:/dev/_spbu/impact_rl/checkpoints/'
        trained_dump_dir = './checkpoints/'
        os.makedirs(trained_dump_dir, exist_ok=True)
        opt_path = trained_dump_dir + cfg_name + '_opt.pth'
        
        # Check if model and optimizer dump exists
        print(opt_path)
        if exists(opt_path):
            print("Models and opt loaded")
            opt_chkp = torch.load(opt_path)
            for adx in range(n_agents):
                #for model_type in all_model_params.keys():
                model.load_state_dict(torch.load(trained_dump_dir + f'{cfg_name}_{adx}_policy_net.pth'), strict=False)
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
                agent_data=agent_data,
                buffer=buffer,
                learning_rate_params=trainer_params['lr_params'],
                dql_params=trainer_params['dql_params'],
                crit_lambdas=trainer_params['losses'],
                device=device,
                cfg_name=cfg_name,
                log_df=log_df,
                log_df_path=train_log_path,
                trained_dump_dir=trained_dump_dir,
                opt_path=opt_path,
                #last_frame_idx=last_frame_idx,
                #min_v_loss=min_v_loss,
                opt_chkp=opt_chkp,
            )
            
