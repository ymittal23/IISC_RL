import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class Bandit:
    def __init__(self, k_arm=10, epsilon=0., UCB_param=None):
        self.k = k_arm
        self.indices = np.arange(self.k)
        self.time = 0
        self.UCB_param = UCB_param
        self.average_reward = 0
        self.epsilon = epsilon

    def reset(self):
        # real reward for each action
        self.q_true = np.random.randn(self.k)

        # estimation for each action
        self.q_estimation = np.zeros(self.k)

        # # of chosen times for each action
        self.action_count = np.zeros(self.k)

        self.best_action = np.argmax(self.q_true) 

    # get an action for this bandit
    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)

        if self.UCB_param is not None:
            UCB_estimation = self.q_estimation + \
                     self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
            q_best = np.max(UCB_estimation)
            return np.random.choice([action for action, q in enumerate(UCB_estimation) if q == q_best])

        q_best = np.max(self.q_estimation)
        return np.random.choice([action for action, q in enumerate(self.q_estimation) if q == q_best])

    # take an action, update estimation for this action
    def step(self, action):
        # generate the reward under N(real reward, 1)
        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.average_reward = (self.time - 1.0) / self.time * self.average_reward + reward / self.time
        self.action_count[action] += 1

        self.q_estimation[action] += 1.0 / self.action_count[action] * (reward - self.q_estimation[action])
        return reward

def simulate(runs, time, bandits):
    best_action_counts = np.zeros((len(bandits), runs, time))
    rewards = np.zeros(best_action_counts.shape)
    for i, bandit in enumerate(bandits):
        for r in tqdm(range(runs)):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1
    best_action_counts = best_action_counts.mean(axis=1)
    rewards = rewards.mean(axis=1)
    return best_action_counts, rewards


def epsilon_method(runs=2000, time=1000):
    epsilons = [0, 0.1, 0.01]
    bandits = [Bandit(epsilon=eps) for eps in epsilons]
    best_action_counts, rewards = simulate(runs, time, bandits)
    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for eps, rewards in zip(epsilons, rewards):
        plt.plot(rewards, label='epsilon = %.02f' % (eps))
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for eps, counts in zip(epsilons, best_action_counts):
        plt.plot(counts, label='epsilon = %.02f' % (eps))
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()
    plt.savefig('/home/affine/RLAssignments/images/epsilon.png')
    plt.close()



def ucb_method(runs=2000, time=1000):
    cValue=[0.5,1,2,5]
    bandits = [Bandit(UCB_param=c) for c in cValue]
    best_action_counts, rewards = simulate(runs, time, bandits)
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    for c, rewards in zip(cValue, rewards):
        plt.plot(rewards, label='C = %.02f' % (c))
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for c, counts in zip(cValue, best_action_counts):
        plt.plot(counts, label='C = %.02f' % (c))
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()
    plt.savefig('/home/affine/RLAssignments/images/ucb.png')
    plt.close()

if __name__ == '__main__':
    epsilon_method()
    ucb_method()
