import argparse
from configparser import ConfigParser

import copy
import os
import torch
# import gym
import slimevolleygym
import numpy as np
import multiprocessing as mp

from Networks.network import Actor, Critic
from sample import SampleAgent
from train import TrainAgent
from evaluationAgent import EvaluationAgent
from metaGameAgent import meta_game

from Utils.utils import make_transition, Dict, RunningMeanStd


def are_parameters_updated(model_before, model_after):
    """
    比较训练前后的模型参数是否更新。

    参数:
    model_before (nn.Module): 训练前的模型。
    model_after (nn.Module): 训练后的模型。

    返回:
    bool: 如果参数更新了返回 True，否则返回 False。
    """
    for param_before, param_after in zip(model_before.parameters(), model_after.parameters()):
        if not torch.equal(param_before, param_after):
            return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")

    parser.add_argument("--algo", type=str, default='ppo', help='algorithm to adjust (default : ppo)')
    parser.add_argument("--process_num", type=int, default=mp.cpu_count(), help="the number of population")
    parser.add_argument("--state_dim", type=int, default=12, help="state_dim")
    parser.add_argument("--action_dim", type=int, default=6, help="action_dim")
    parser.add_argument("--env", type=int, default=slimevolleygym.SlimeVolleyEnv(), help="environment")

    parser.add_argument("--max_step_per_episode", type=int, default=3000, help="max_step_per_episode")
    parser.add_argument("--mini_batch_num", type=int, default=10, help="split batch into 10 part")

    parser.add_argument("--num_rollout", type=int, default=10, help="PPO parameter")
    parser.add_argument("--eval_count", type=int, default=100, help="evaluation count")
    parser.add_argument("--sample_proportion_mode", type=int, default=1,
                        help="sample_proportion mode 1: SP 2: Uniform Distribution 3: nash")

    parser.add_argument("--lr_meta", type=float, default=5e-3, help="Learning rate of sample proportion")
    parser.add_argument("--battle_episodes_for_winning_rate_matrix", type=int, default=100,
                        help="battle_episodes_for_winning_rate_matrix")
    parser.add_argument("--winning_rate_threshold_for_policy_improvement", type=int, default=0.9,
                        help="winning_rate_threshold_for_policy_improvement")

    parser.add_argument("--max_winning_rate", type=float, default=0.9, help="max_winning_rate")
    parser.add_argument("--max_actor_training_num", type=float, default=1000, help="max_winning_rate")
    parser.add_argument("--reward_scaling", type=float, default=0.1, help='reward scaling(default : 0.1)')

    args = parser.parse_args()
    parser = ConfigParser()
    parser.read('config.ini')
    agent_args = Dict(parser, args.algo)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor_pop = [Actor(agent_args.layer_num,
                       args.state_dim,
                       args.action_dim,
                       agent_args.hidden_dim,
                       agent_args.activation_function,
                       agent_args.last_activation,
                       agent_args.trainable_std
                       )]
    critic_pop = [Critic(agent_args.layer_num,
                         args.state_dim,
                         1,
                         agent_args.hidden_dim,
                         agent_args.activation_function,
                         agent_args.last_activation
                         )]

    sample_proportion = np.array([1.])

    while True:
        actor_training = Actor(agent_args.layer_num,
                               args.state_dim,
                               args.action_dim,
                               agent_args.hidden_dim,
                               agent_args.activation_function,
                               agent_args.last_activation,
                               agent_args.trainable_std
                               )
        critic_training = Critic(agent_args.layer_num,
                                 args.state_dim,
                                 1,
                                 agent_args.hidden_dim,
                                 agent_args.activation_function,
                                 agent_args.last_activation
                                 )
        flag = 0
        state_i_lst, state_j_lst = [], []
        state_rms_i = RunningMeanStd(args.state_dim)
        state_rms_j = RunningMeanStd(args.state_dim)
        for n_epi in range(args.max_actor_training_num):
            print("#######################################################")
            print("generation: " + str(len(actor_pop)))
            # sample
            print("sampling...")
            sampleAgent = SampleAgent(args, actor_pop, critic_pop, actor_training, critic_training, sample_proportion,
                                      agent_args, device)
            buffer_i, buffer_j = sampleAgent.sample(state_i_lst, state_rms_i, state_j_lst, state_rms_j)
            # training
            print("training...")
            trainAgent = TrainAgent(args, actor_training, critic_training, buffer_i, agent_args, device)
            trainAgent.train(n_epi)
            state_rms_i.update(np.vstack(state_i_lst))
            state_rms_j.update(np.vstack(state_j_lst))
            evaluationAgent = EvaluationAgent(args, actor_training, actor_pop, critic_training, critic_pop,
                                              sample_proportion, agent_args, device)
            print("evaluating...")
            winning_rate = evaluationAgent.evaluation(state_rms_i, state_rms_j)
            print("winning rate: " + str(winning_rate))
            print("#######################################################")
            flag = n_epi
            if winning_rate > args.max_winning_rate:
                actor_pop.append(actor_training)
                critic_pop.append(critic_training)
                sample_proportion = meta_game(args, actor_pop, critic_pop, sample_proportion)
                break
        if flag == args.max_actor_training_num - 1:
            print("training finished")
            break
