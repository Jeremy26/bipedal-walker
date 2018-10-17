import os
import numpy as np
import gym
from gym import wrappers
# import pybullet_envs


class Normalizer():
    # Normalizes the inputs
    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std

## Algorithm

class Walker():
    def __init__(self,nb_steps=1000, episode_length=2000, learning_rate=0.02, num_deltas=16, num_best_deltas=16, noise=0.03, seed=1000, env_name='BipedalWalker-v2',record_every=50, monitor_dir = None):
        self.nb_steps = nb_steps
        self.episode_length = episode_length
        self.learning_rate = learning_rate
        self.num_deltas = num_deltas
        self.num_best_deltas = num_best_deltas
        assert self.num_best_deltas <= self.num_deltas
        self.noise = noise
        self.seed = seed
        self.record_every = record_every
        np.random.seed(self.seed)
        self.env = gym.make(env_name)
        if monitor_dir is not None:
            should_record = lambda i: self.record_video
            self.env = wrappers.Monitor(self.env, monitor_dir, video_callable=should_record, force=True)
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.shape[0]
        self.episode_length = self.env.spec.timestep_limit
        self.record_video = False
        self.theta = np.zeros((self.output_size, self.input_size))
        self.normalizer = Normalizer(self.input_size)
        
    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(self.num_deltas)]
    
    def evaluate(self, input, delta = None, direction = None):
        if direction is None:
            return self.theta.dot(input)
        elif direction == "+":
            return (self.theta + self.noise * delta).dot(input)
        elif direction == "-":
            return (self.theta - self.noise * delta).dot(input)
    
    def play_episode(self, direction=None, delta=None):
        state = self.env.reset()
        done = False
        num_plays = 0.0
        sum_rewards = 0.0
        while not done and num_plays < self.episode_length:
            self.normalizer.observe(state)
            state = self.normalizer.normalize(state)
            action = self.evaluate(state,delta,direction)
            state, reward, done, _ = self.env.step(action)
            reward = max(min(reward, 1), -1)
            sum_rewards += reward
            num_plays += 1
        return sum_rewards
        
    def train(self):
        for iteration in range(self.nb_steps):
            # Generate num_deltas deltas and evaluate positive and negative rewards
            deltas = self.sample_deltas()
            positive_rewards = [0] * self.num_deltas
            negative_rewards = [0] * self.num_deltas

            # Run num_deltas episode with positive and negative variations
            for i in range(self.num_deltas):
                positive_rewards[i] = self.play_episode(direction="+",delta=deltas[i])
                negative_rewards[i] = self.play_episode(direction="-",delta=deltas[i])

            # Collect rollouts r+,r-,delta 
            rollouts = zip(positive_rewards, negative_rewards, deltas)

            # Calculate the standard deviation of all the rewards
            sigma_rewards = np.array(positive_rewards + negative_rewards).std()

            # Sort the rollouts by maximum reward and select best_num_deltas rollouts
            scores = {k:max(r_pos, r_neg) for k,(r_pos,r_neg) in enumerate(zip(positive_rewards,negative_rewards))}
            order = sorted(scores.keys(), key = lambda x:scores[x])[:self.num_best_deltas]
            rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

            # Calculate step
            step = np.zeros(self.theta.shape)
            for pos, neg, d in rollouts:
                step += (pos-neg)*d

            # Update the weights
            self.theta += self.learning_rate/(self.num_best_deltas*sigma_rewards*step)

            # Only record video during evaluation, every n steps
            if iteration % self.record_every == 0:
                self.record_video = True

            # Play an episode with the new weights and see improvement
            final_reward = self.play_episode() ## We play without + or - noise
            print('Step: ', iteration, 'Reward: ', final_reward)

            self.record_video = False


def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# Main code
if __name__ == '__main__':
    ENV_NAME = "BipedalWalker-v2"
    videos_dir = mkdir('.', 'videos')
    monitor_dir = mkdir(videos_dir, ENV_NAME)
    trainer = Walker(monitor_dir=monitor_dir)
    trainer.train()
