import numpy as np


def calculate_payoffs(args, actor_pop, num_matches=10):
    num_actors = len(actor_pop)
    payoffs = np.zeros((num_actors, num_actors))
    for i in range(num_actors):
        for j in range(num_actors):
            if i != j:
                total_score_i = 0
                total_score_j = 0
                for _ in range(num_matches):
                    args.env.reset()
                    done = False
                    while not done:
                        action_i = actor_pop[i].select_action(args.env)
                        action_j = actor_pop[j].select_action(args.env)
                        _, reward, done, _ = args.env.step([action_i, action_j])
                        total_score_i += reward[0]
                        total_score_j += reward[1]
                payoffs[i][j] = total_score_i / num_matches  # 计算 actor i 的平均得分
    return payoffs


def meta_game(args, actor_pop, critic_pop, sample_proportion):
    if args.sample_proportion_mode == 1:
        sample_proportion = np.insert(sample_proportion, 0, 0)
    elif args.sample_proportion_mode == 2:
        # 均匀分布
        n = len(actor_pop)
        sample_proportion = np.full(n, 1/n)
    else:
        # 纳什均衡
        pass
    return sample_proportion
