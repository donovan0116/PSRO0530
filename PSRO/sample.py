import numpy as np
import torch
from torch.distributions import Categorical

from Utils.utils import make_transition, RunningMeanStd

'''
思路：构建sample的agent，采样完成后将data传出去，再在train中创建agent进行训练。
'''
from PSRO.ppo import PPO


class SampleAgent:
    def __init__(self, args, actor_pop, critic_pop, actor_training, critic_training, sample_proportion, agent_args,
                 device):
        self.traj_length = agent_args.traj_length
        self.batch_size = agent_args.batch_size
        self.mini_batch_num = args.mini_batch_num
        self.num_rollout = args.num_rollout
        self.device = device

        self.agent_args = agent_args

        self.state_dim = args.state_dim
        self.action_dim = args.action_dim

        self.env = args.env
        self.actor_pop = actor_pop
        self.critic_pop = critic_pop
        self.actor_training = actor_training
        self.critic_training = critic_training
        self.sample_proportion = sample_proportion
        self.reward_scaling = args.reward_scaling

        self.action_table = [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0]]

    def sample(self, state_i_lst, state_rms_i, state_j_lst, state_rms_j):
        # 通过sample_proportion选择actor_pop
        sample_pro = torch.from_numpy(self.sample_proportion).to(self.device).float().detach()
        sample_num = Categorical(sample_pro).sample().detach().numpy().tolist()
        score_lst = []

        agent_i = PPO(self.device, self.state_dim, self.action_dim, self.agent_args,
                      self.actor_training, self.critic_training)

        agent_j = PPO(self.device, self.state_dim, self.action_dim, self.agent_args,
                      self.actor_pop[sample_num], self.critic_pop[sample_num])

        score = 0
        state_i_ = self.env.reset()
        # state_i_ = state_i_[0]
        state_j_ = state_i_
        state_i = np.clip((state_i_ - state_rms_i.mean) / (state_rms_i.var ** 0.5 + 1e-8), -5, 5)
        state_j = np.clip((state_j_ - state_rms_j.mean) / (state_rms_j.var ** 0.5 + 1e-8), -5, 5)
        for t in range(self.traj_length):
            state_i_lst.append(state_i_)
            state_j_lst.append(state_j_)

            mu_i, sigma_i = agent_i.get_action(
                torch.from_numpy(np.array(state_i)).float().to(self.device).unsqueeze(dim=0))
            mu_j, sigma_j = agent_j.get_action(
                torch.from_numpy(np.array(state_j)).float().to(self.device).unsqueeze(dim=0))

            dist_i = torch.distributions.Normal(mu_i, sigma_i[0])
            dist_j = torch.distributions.Normal(mu_j, sigma_j[0])

            action_i = dist_i.sample()
            action_j = dist_j.sample()

            log_prob_i = dist_i.log_prob(action_i).sum(-1, keepdim=True)
            log_prob_j = dist_j.log_prob(action_j).sum(-1, keepdim=True)

            action_i = action_i.detach().numpy().tolist()
            action_i = action_i[0]
            action_j = action_j.detach().numpy().tolist()
            action_j = action_j[0]
            action_i_ = self.action_table[action_i.index(max(action_i))]
            action_j_ = self.action_table[action_j.index(max(action_j))]
            # next_state_i_, reward_i, done, info = self.env.step(action_i_)
            next_state_i_, reward_i, done, info = self.env.step(action_i_, action_j_)
            next_state_j_ = info["otherObs"]
            next_state_i = np.clip((next_state_i_ - state_rms_i.mean) / (state_rms_i.var ** 0.5 + 1e-8), -5, 5)
            next_state_j = np.clip((next_state_j_ - state_rms_j.mean) / (state_rms_j.var ** 0.5 + 1e-8), -5, 5)
            transition_i = make_transition(state_i,
                                           # action_i.cpu().numpy(),
                                           action_i,
                                           np.array([reward_i * self.reward_scaling]),
                                           next_state_i,
                                           np.array([done]),
                                           log_prob_i.detach().cpu().numpy()
                                           )
            agent_i.put_data(transition_i)
            score += reward_i
            if reward_i == 1.01:
                reward_j = -0.99
            elif reward_i == -0.99:
                reward_j = 1.01
            else:
                reward_j = 0.01
            transition_j = make_transition(state_j,
                                           action_j,
                                           np.array([reward_j * self.reward_scaling]),
                                           next_state_j,
                                           np.array([done]),
                                           log_prob_j.detach().cpu().numpy()
                                           )
            agent_j.put_data(transition_j)
            if done:
                state_i_ = self.env.reset()
                # state_i_ = state_i_[0]
                state_j_ = state_i_
                state_i = np.clip((state_i_ - state_rms_i.mean) / (state_rms_i.var ** 0.5 + 1e-8), -5, 5)
                state_j = np.clip((state_j_ - state_rms_j.mean) / (state_rms_j.var ** 0.5 + 1e-8), -5, 5)
                score_lst.append(score)
                score = 0
            else:
                state_i = next_state_i
                state_j = next_state_j
                state_i_ = next_state_i_
                state_j_ = next_state_j_

        return agent_i.data, agent_j.data
