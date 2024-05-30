from PSRO.ppo import PPO


class TrainAgent:
    def __init__(self, args, actor_training, critic_training, buffer, agent_args, device):
        self.agent_args = agent_args
        self.device = device

        self.state_dim = args.state_dim
        self.action_dim = args.action_dim

        self.actor_training = actor_training
        self.critic_training = critic_training
        self.buffer = buffer

    def train(self, n_epi):
        agent = PPO(self.device, self.state_dim, self.action_dim, self.agent_args,
                    self.actor_training, self.critic_training)
        agent.data = self.buffer

        agent.train_net(n_epi)
