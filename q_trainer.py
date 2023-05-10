import os
import numpy as np
import pandas as pd
import time
import collections

import torch
from torch import nn
from torch.optim import Adam



def init_weights_xavier(m):
    """Xavier weight initializer."""
    if type(m) == (nn.Conv2d or nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        #m.bias.data.fill_(0.01)


def init_weights_kaiming(m):
    """Kaiming weight initializer."""
    if type(m) == (nn.Conv2d or nn.Conv1d):
        nn.init.kaiming_normal_(m.weight)
        #m.bias.data.fill_(0.01)


def fft_l1_norm(fft_size=4096):
    def get_norm(data, rec_data):
        l = data.shape[-1]
        pad_size = fft_size - l
        pad_left = pad_size // 2
        pad_right = pad_size - pad_left
        data_padded = nn.functional.pad(data, (pad_left, pad_right))
        data_fft = torch.abs(torch.fft.fft(data_padded))
        norm_value = nn.functional.smooth_l1_loss(data_fft, torch.zeros_like(data_fft).to(data_fft.device))
        return norm_value
    return get_norm


def train_q_agent(agent_data, 
                  learning_rate_params,
                  dql_params,
                  crit_lambdas, 
                  device, 
                  run_name,
                  log_df_path, trained_dump_dir, opt_path,
                  last_frame_idx=0,
                  min_v_loss=np.Inf,
                  opt_chkp=None):
    """
    The trainer function for agent training.

    Args:
        agent_data:             Agent data (Dictionary):
            agent:              Agent class
            policy_net:         Policy Q-network
            target_net:         Target Q-network
        learning_rate_params:   Learning rate params for optimizer and scheduler
        dql_params:             Deep Q-Learning parameters (eps greedy parameters, replay buffer, etc.)
        crit_lambdas:           Weight coefficients for loss functions
        device:                 Current device (cuda or cpu)
        run_name:               Name of the experiment for logging
        log_df_path:            Path to training logs
        trained_dump_path:      Path to model dump
        opt_path:               Path to the optimizer and scheduler checkpoints
        last_frame_idx:         Last frame index before current transfer learning
        min_v_loss:             Minimum validation loss among all frame_idxs
        opt_chkp:               Optimizer and scheduler checkpoints
    """

    # Loss and optimizer
    crits = {
        'l2': nn.MSELoss(),
        'l1': nn.L1Loss(),
        'l1S': nn.SmoothL1Loss(),
        'l1norm': fft_l1_norm(),
    }             
    
    agent = agent_data['agent']
    buffer = agent.exp_buffer
    policy_net = agent_data['policy_net']
    target_net = agent_data['target_net']
    #model_params = []
    #for _, model in models.values():
    #    model_params += list(model.parameters())
    optimizer = Adam(policy_net.parameters(), lr=learning_rate_params['lr'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, 
                                                   step_size=learning_rate_params['sched_step'], 
                                                   gamma=learning_rate_params['sched_gamma'])

    # Checkpoint processing
    optimizer_chkpt_name = 'optimizer'
    scheduler_chkpt_name = 'scheduler'
    if opt_chkp:
        print("Optimizer and Scheduler restored")
        optimizer.load_state_dict(opt_chkp[optimizer_chkpt_name])
        lr_scheduler.load_state_dict(opt_chkp[scheduler_chkpt_name])

    # Training
    eps_start = dql_params['eps_start']
    eps_decay = dql_params['eps_decay']
    eps_min = dql_params['eps_min']
    dql_gamma = dql_params['dql_gamma']
    sync_target_frames = dql_params['sync_target_frames']
    replay_start_size = dql_params['replay_start_size']
    batch_size = dql_params['batch_size']
    n_trials = dql_params['n_trials']
    total_rewards_size = dql_params['total_rewards_size']

    best_mean_reward = None
    total_rewards = collections.deque(maxlen=total_rewards_size)
    loss_vals = {}
    frame_idx = last_frame_idx
    trial_idx = 0
    epsilon = eps_start
    
    if last_frame_idx == 0:
        log_header = True
    else:
        log_header = False

    start_t = time.time() 

    while True:
        frame_idx += 1

        for lm in crit_lambdas.keys():
            loss_vals[lm] = 0.

        epsilon = max(epsilon * eps_decay, eps_min)
        reward = agent.play_step(policy_net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            mean_reward = np.mean(total_rewards)

            print("%d:  %d games, mean reward %.3f, (epsilon %.2f)" % (
                frame_idx, trial_idx, mean_reward, epsilon))
            
            if best_mean_reward is None or mean_reward > best_mean_reward:
                torch.save(policy_net.state_dict(), trained_dump_dir + f'{run_name}_0_policy_net.pth')
                torch.save({
                    optimizer_chkpt_name: optimizer.state_dict(),
                    scheduler_chkpt_name: lr_scheduler.state_dict()
                }, opt_path)
                best_mean_reward = mean_reward
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f" % (best_mean_reward)) 

            if trial_idx > n_trials:
                print("Solved in %d frames!" % frame_idx)
                break
            
            trial_idx += 1

        if len(buffer) < replay_start_size:
            continue

        batch = buffer.sample(batch_size)
        states, actions, rewards, dones, next_states = batch

        states_v = torch.tensor(states).to(device)
        next_states_v = torch.tensor(next_states).to(device)
        actions_v = torch.LongTensor(actions).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        done_mask = torch.BoolTensor(dones).to(device)

        next_state_values = target_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()
        expected_state_action_values = next_state_values * dql_gamma + rewards_v
        state_action_values = policy_net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        #state_action_values = policy_net(states_v).gather(1, actions_v)

        losses = {}
        for lm in crit_lambdas.keys():
            losses[lm] = crit_lambdas[lm] * crits[lm](state_action_values, expected_state_action_values)
            loss_vals[lm] += losses[lm].cpu().item() # / n_train_batches

        optimizer.zero_grad()
        loss = sum(losses.values())
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if loss_vals[list(crit_lambdas.keys())[0]] < min_v_loss:
            min_v_loss = loss_vals[list(crit_lambdas.keys())[0]]

        if frame_idx % sync_target_frames == 0:
            #print(agent.env.get_img_state())
            print(agent.env.state)
            print(agent.env.target_state)
            d = {"frame_idx": [frame_idx], "min_v_loss": [min_v_loss]}
            for lm in crit_lambdas.keys():
                d[lm] = [loss_vals[lm]]
                loss_vals[lm] = 0. 
            d_rounded = {key: round(value[0], 7) for key, value in d.items()}
            print(d_rounded)
            df = pd.DataFrame.from_dict(d)
            df.to_csv(log_df_path, mode='a', header=log_header, index=False)
            log_header = False

            target_net.load_state_dict(policy_net.state_dict())

    print("t: ", time.time() - start_t)




