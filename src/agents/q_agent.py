import numpy as np
import torch
import collections
import itertools



Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Simple_Agent: 

    id_iter = itertools.count()

    def __init__(self, env):
        self.id = next(Simple_Agent.id_iter)
        self.env = env
        self._reset()
    
    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device='cpu'):    
        done_reward = None    
        #print("Mask", self.env.action_space.legal_action_mask)
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
            self.env.action_space.adjust_legal(action)
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            q_vals_v[torch.tensor(self.env.action_space.legal_action_mask == 0).view(1, -1)] = -float('inf')
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())
            self.env.action_space.adjust_legal(action)

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        
        experience = Experience(self.state, action, reward, is_done, new_state)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
            return done_reward, experience
        else:
            return None, experience

