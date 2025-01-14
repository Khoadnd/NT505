import random
import math
from statistics import mean
from collections import deque
import pickle
from gym_malware.envs.controls import manipulate2 as manipulate
from gym_malware.envs.utils import interface, pefeatures
import gym_malware
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import joblib
import numpy as np
import gym
from collections import namedtuple, deque
from pyfiglet import Figlet
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich import print
from rich.traceback import install
from rich.progress import Progress, TaskID, track
from rich.logging import RichHandler
import os
from datetime import date
from tqdm import tqdm
from pathlib import Path
from itertools import count
import argparse
from logging import basicConfig, exception, debug, error, info, warning, getLogger
import logging
import re
import warnings
warnings.filterwarnings("ignore")


np.random.seed(123)
# from sklearn.externals import joblib


def put_banner():
    # Printing heading banner
    f = Figlet(font="banner4")
    grid = Table.grid(expand=True, padding=1, pad_edge=True)
    grid.add_column(justify="right", ratio=38)
    grid.add_column(justify="left", ratio=62)
    grid.add_row(
        Text.assemble((f.renderText("PE"), "bold red")),
        Text(f.renderText("Sidious"), "bold white"),
    )
    print(grid)
    print(
        Panel(
            Text.assemble(
                ("Creating Chaos with Mutated Evasive Malware with ", "grey"),
                ("Reinforcement Learning ", "bold red"),
                ("and "),
                ("Generative Adversarial Networks", "bold red"),
                justify="center",
            )
        )
    )


put_banner()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Reinforcement Training Module')

    parser.add_argument('--rl_gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')

    parser.add_argument('--rl_episodes', type=float, default=1000,
                        help='number of episodes to execute (default: 1000)')
    parser.add_argument('--con', type=str, default=None, required=True,
                        help='Continued train model')
    parser.add_argument('--model', type=Path, default=None,
                        help='Specify model for continue training')
    parser.add_argument('--start', type=int, required=True,
                        help='number of episodes to execute (default: 1000)')
    parser.add_argument('--rl_mutations', type=float, default=80,
                        help='number of maximum mutations allowed (default: 80)')
    parser.add_argument('--classifier', type=Path, required=True,
                        help='End index')
    parser.add_argument('--sections', type=Path, required=True,
                        help='End index')
    parser.add_argument('--imports', type=Path, required=True,
                        help='End index')

    parser.add_argument('--rl_save_model_interval', type=float, default=200, required=True,
                        help='Interval at which models should be saved (default: 500)')  # gitul
    parser.add_argument('--rl_output_directory', type=Path, default=Path("models"),
                        help='Path to save the models in (default: models)')  # gitul

    parser.add_argument("--logfile", help="The file path to store the logs. (default : rl_features_logs_" +
                        str(date.today()) + ".log)", type=Path, default=Path("rl_features_logs_" + str(date.today()) + ".log"))
    logging_level = ["debug", "info", "warning", "error", "critical"]
    parser.add_argument(
        "-l",
        "--log",
        dest="log",
        metavar="LOGGING_LEVEL",
        choices=logging_level,
        default="info",
        help=f"Select the logging level. Keep in mind increasing verbosity might affect performance. Available choices include : {logging_level}",
    )

    args = parser.parse_args()
    return args


def logging_setup(logfile: str, log_level: str):

    from imp import reload
    reload(logging)

    log_dir = "Logs"

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    logfile = os.path.join(log_dir, logfile)

    basicConfig(
        level=log_level.upper(),
        filemode='a',  # other options are w for write.
        format="%(message)s",
        filename=logfile
    )

    getLogger().addHandler(RichHandler())

    info("[*] Starting Reinforcement Learning Agent's Training ...\n")


args = parse_args()
logging_setup(str(args.logfile), args.log)

info("[*] Initilializing environment ...\n")
interface.local_model = pickle.load(open(str(args.classifier), 'rb'))
# interface.local_model = joblib.load(open(str(args.classifier), 'rb'))
manipulate.COMMON_IMPORTS = pickle.load(open(str(args.imports), 'rb'))
manipulate.COMMON_SECTION_NAMES = pickle.load(open(str(args.sections), 'rb'))

env_id = "malware-score-v0"
env = gym.make(env_id)
env.seed(123)

np.random.seed(123)

ACTION_LOOKUP = {i: act for i, act in enumerate(
    manipulate.ACTION_TABLE.keys())}

device = torch.device("cpu")

USE_CUDA = False
Variable = lambda *args, **kwargs: autograd.Variable(
    *args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


# prioritized replay buffer
class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):

        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            e = self.experience(state, action, reward, next_state, done)
            self.buffer.append(e)
        else:
            e = self.experience(state, action, reward, next_state, done)
            self.buffer[self.pos] = e

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones, indices)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


def update_epsilon(n):
    epsilon_start = 1.0
    epsilon = epsilon_start
    epsilon_final = 0.4
    epsilon_decay = 1000  # N from the research paper (equation #6)

    epsilon = 1.0 - (n/epsilon_decay)

    if epsilon <= epsilon_final:
        epsilon = epsilon_final

    return epsilon

# create a dqn class


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, env.action_space.n)
        )

    def forward(self, x):
        return self.layers(x)

    def chooseAction(self, observation, epsilon):
        rand = np.random.random()
        if rand > epsilon:
            # observation = torch.from_numpy(observation).float().unsqueeze(0).to(device)
            actions = self.forward(observation)
            action = torch.argmax(actions).item()

        else:
            action = np.random.choice(env.action_space.n)

        return action


replay_buffer = NaivePrioritizedBuffer(500000)

info("[*] Initilializing Neural Network model ...")

if args.con == 'yes':
    current_model = DQN().to(device)
    current_model.load_state_dict(torch.load(str(args.model)))
    current_model.eval()
    target_model = DQN().to(device)
elif args.con == 'no':
    current_model = DQN().to(device)
    target_model = DQN().to(device)

optimizer = optim.Adam(current_model.parameters())

gamma = 0.99  # discount factor as mentioned in the paper


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

# TD loss


def compute_td_loss(batch_size):
    state, action, reward, next_state, done, indices = replay_buffer.sample(
        batch_size, 0.4)

    Q_targets_next = target_model(next_state).detach().max(1)[0].unsqueeze(1)
    Q_targets = reward + (gamma * Q_targets_next * (1 - done))
    Q_expected = current_model(state).gather(1, action)

    loss = (Q_expected - Q_targets.detach()).pow(2)
    prios = loss + 1e-5
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    optimizer.step()

    return loss


# normaliza the features
class RangeNormalize(object):
    def __init__(self,
                 min_val,
                 max_val):
        """
        Normalize a tensor between a min and max value
        Arguments
        ---------
        min_val : float
                lower bound of normalized tensor
        max_val : float
                upper bound of normalized tensor
        """
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            _min_val = _input.min()
            _max_val = _input.max()
            a = (self.max_val - self.min_val) / (_max_val - _min_val)
            b = self.max_val - a * _max_val
            _input = (_input * a) + b
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]


def main():
    info("[*] Starting training ...")
    D = int(args.rl_episodes)
    T = int(args.rl_mutations)
    # as mentioned in the paper (number of steps before learning starts)
    B = 1000
    batch_size = 32  # as mentioned in the paper (batch_size)
    losses = []
    reward_ben = 20
    n = 0  # current training step
    rn = RangeNormalize(-0.5, 0.5)
    check = False

    for episode in range(args.start, D):
        state = env.reset()
        state_norm = rn(state)
        state_norm = torch.from_numpy(
            state_norm).float().unsqueeze(0).to(device)
        for mutation in range(1, T):
            n = n + 1
            epsilon = update_epsilon(n)
            action = current_model.chooseAction(state_norm, epsilon)
            next_state, reward, done, _ = env.step(action)
            debug("\t[+] Episode : " + str(episode) + " , Mutation # : " + str(mutation) +
                  " , Mutation : " + str(ACTION_LOOKUP[action]) + " , Reward : " + str(reward))
            next_state_norm = rn(next_state)
            next_state_norm = torch.from_numpy(
                next_state_norm).float().unsqueeze(0).to(device)

            if reward == 10.0:
                power = -((mutation-1)/T)
                reward = (math.pow(reward_ben, power))*100

            replay_buffer.push(state_norm, action, reward,
                               next_state_norm, done)

            if len(replay_buffer) > B:
                loss = compute_td_loss(batch_size)
                losses.append(loss.item())

            if done:
                break

            state_norm = next_state_norm

        debug('\t[+] Episode Over')
        if n % 100 == 0:
            update_target(current_model, target_model)

        if episode % args.rl_save_model_interval == 0:
            if not os.path.exists(args.rl_output_directory):
                os.mkdir(args.rl_output_directory)
                info("[*] model directory has been created at : " +
                     str(args.rl_output_directory))
            torch.save(current_model.state_dict(), os.path.join(
                args.rl_output_directory, "rl-model-" + str(episode) + "-" + str(date.today()) + ".pt"))
            info("[*] Saving model in models/ directory ...")

    torch.save(current_model.state_dict(), os.path.join(
        args.rl_output_directory, "rl-model-" + str(D) + "-" + str(date.today()) + ".pt"))
    info("[*] Saving model in models/ directory ...")


if __name__ == '__main__':
    main()
