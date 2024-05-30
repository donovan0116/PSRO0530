import numpy as np
import torch
from torch.distributions import Categorical

from PSRO.ppo import PPO


class EvaluationAgent:
    def __init__(self, args, actor_training, actor_pop, critic_training, critic_pop, sample_proportion, agent_args,
                 device):
        self.actor_training = actor_training
        self.actor_pop = actor_pop
        self.sample_proportion = sample_proportion

        self.traj_length = agent_args.traj_length
        self.env = args.env
        self.eval_count = args.eval_count
        self.device = device
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim

        self.agent_args = agent_args
        self.critic_training = critic_training
        self.critic_pop = critic_pop

        self.action_table = [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0]]

    def evaluation(self, state_rms_i, state_rms_j):
        # 通过sample_proportion选择actor_pop
        sample_pro = torch.from_numpy(self.sample_proportion).to(self.device).float().detach()
        sample_num = Categorical(sample_pro).sample().detach().numpy().tolist()
        win_count = 0

        agent_i = PPO(self.device, self.state_dim, self.action_dim, self.agent_args,
                      self.actor_training, self.critic_training)

        agent_j = PPO(self.device, self.state_dim, self.action_dim, self.agent_args,
                      self.actor_pop[sample_num], self.critic_pop[sample_num])

        for _ in range(self.eval_count):
            state_i_ = self.env.reset()
            # state_i_ = state_i_[0]
            state_j_ = state_i_
            state_i = np.clip((state_i_ - state_rms_i.mean) / (state_rms_i.var ** 0.5 + 1e-8), -5, 5)
            state_j = np.clip((state_j_ - state_rms_j.mean) / (state_rms_j.var ** 0.5 + 1e-8), -5, 5)
            tot_reward_i, tot_reward_j = 0, 0
            for step in range(self.traj_length):
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
                tot_reward_i += reward_i
                next_state_j_ = info["otherObs"]
                next_state_i = np.clip((next_state_i_ - state_rms_i.mean) / (state_rms_i.var ** 0.5 + 1e-8), -5, 5)
                next_state_j = np.clip((next_state_j_ - state_rms_j.mean) / (state_rms_j.var ** 0.5 + 1e-8), -5, 5)
                if reward_i == 1.01:
                    reward_j = -0.99
                elif reward_i == -0.99:
                    reward_j = 1.01
                else:
                    reward_j = 0.01
                tot_reward_j += reward_j
                if done or step == self.traj_length - 1:
                    # 如果i胜利，则存入win_count + 1
                    if tot_reward_i >= tot_reward_j:
                        win_count += 1
                    break
                else:
                    state_i = next_state_i
                    state_j = next_state_j
                    state_i_ = next_state_i_
                    state_j_ = next_state_j_

        return win_count / self.eval_count
