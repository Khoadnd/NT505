from urllib.request import Request
import warnings
warnings.filterwarnings("ignore")
import math, random

import gym
import numpy as np
import sys
import os
np.random.seed(123)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd 

#import gym_malware
from gym_malware.envs.utils import interface, pefeatures2
from gym_malware.envs.controls import manipulate2 as manipulate
# from sklearn.externals import joblib
from collections import namedtuple, deque
from statistics import mean 

import argparse
import logging
from logging import basicConfig, exception, debug, error, info, warning, getLogger
from rich.logging import RichHandler
from rich.progress import Progress, TaskID, track
from rich.traceback import install

from rich import print
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from pyfiglet import Figlet

from pathlib import Path
from tqdm import tqdm
from datetime import date
import os

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

# put_banner()

#env_id = "malware-score-v0"
#env = gym.make(env_id)
#env.seed(123)
device = torch.device("cpu")

from collections import deque

def parse_args():
	parser = argparse.ArgumentParser(description='Testing Module')

	parser.add_argument('-f', type=Path, help='Path to input file')

	parser.add_argument('-d', type=Path, 
						help='Path to input directory')

	parser.add_argument('-o', type=Path, default=Path('Mutated_malware/'), required=True
						,help='Path to output directory (default : Mutated_malware/)')

	parser.add_argument('--saved_model', type=Path, default=Path('models/rl-model.pt'), required=True,
						help='Path to saved model')
	
	#parser.add_argument('--start', type=int, required=True,
	#					help='Start index')
	# parser.add_argument('--list_files', type=Path, required=True,
	#					help='Start index')
	
	#parser.add_argument('--end', type=int, required=True,
	#					help='End index')
	parser.add_argument('--sections', type=Path, required=True,
						help='End index')
	parser.add_argument('--imports', type=Path, required=True,
						help='End index')
	parser.add_argument('--classifier', type=Path, required=True,
						help='End index')

	parser.add_argument('--rl_mutations', type=int, default=80,
					help='number of maximum mutations allowed (default: 80)')

	parser.add_argument("--logfile", help = "The file path to store the logs. (default : rl_test_" + str(date.today()) + ".log)", type = Path, default = Path("rl_test_logs_" + str(date.today()) + ".log"))
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

args = parse_args()
import pickle
interface.local_model = pickle.load(open(str(args.classifier), 'rb'))
# interface.local_model = joblib.load(open(str(args.classifier), 'rb'))
manipulate.COMMON_IMPORTS = pickle.load(open(str(args.imports), 'rb'))
manipulate.COMMON_SECTION_NAMES = pickle.load(open(str(args.sections), 'rb'))
ACTION_LOOKUP = {i: act for i, act in enumerate(manipulate.ACTION_TABLE.keys())}

def logging_setup(logfile: str , log_level: str):

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

# create a dqn class
class DQN(nn.Module):
	def __init__(self):
		super(DQN, self).__init__()
		self.layers = nn.Sequential(
			nn.Linear(2350, 256),
			nn.ReLU(),
			nn.Linear(256, 64),
			nn.ReLU(),
			nn.Linear(64, len(manipulate.ACTION_TABLE.keys()))
		)

	def forward(self, x):
		return self.layers(x)

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
			b = self.max_val- a * _max_val
			_input = (_input * a ) + b
			outputs.append(_input)
		return outputs if idx > 1 else outputs[0]

def load_model(saved_model):
	#from rl_train import Policy
	model = DQN().to(device)
	model.load_state_dict(torch.load(str(saved_model)))
	model.eval()
	return model

def generate_mutated_malware(file, model, args):

	pe = pefeatures2.PEFeatureExtractor2()
	rn = RangeNormalize(-0.5,0.5)

	info("[*] Reading file : " + str(file))
	bytez = []
	number_of_action_mutate = args.rl_mutations
	with open(str(file), 'rb') as infile:
		bytez = infile.read()

	for t in track(range(1, args.rl_mutations) , description="Generating mutation ...", transient=True):
		state = pe.extract( bytez )
		state_norm = rn(state)
		state_norm = torch.from_numpy(state_norm).float().unsqueeze(0).to(device)
		
		actions = model.forward(state_norm)
		action = torch.argmax(actions).item()
		action = ACTION_LOOKUP[action]
		
		bytez = bytes(manipulate.modify_without_breaking(bytez, [action]))
		
		new_score = interface.get_score_local( bytez )

		if(new_score < interface.local_model_threshold):
			number_of_action_mutate = t
			break
		del state, state_norm, actions, action, new_score

	if not os.path.exists(args.o):
		os.mkdir(args.o)
		info("[*] output directory has been created at : " + str(args.o))
	output_file = os.path.join(args.o, "mutated_" + str(os.path.basename(file)) + '_' + str(number_of_action_mutate))
	info("[*] Writing mutated file to : " + str(output_file) + "\n\n")
	with open(str(output_file), mode='wb') as file1:
		file1.write(bytes(bytez))
		return

def main():

	
	logging_setup(str(args.logfile), args.log)

	info("[*] Loading model : " + str(args.saved_model))
	model = load_model(args.saved_model)

	if(args.f):
		generate_mutated_malware(args.f , model, args)
	elif(args.d):
		# with open(str(args.list_files), 'r') as f:
		# 	mutated_list = f.read().split('\n')[:-1]
		
		dirs = os.listdir(args.d)
		for i in range(len(dirs)):
			dirs[i] = os.path.join(args.d, dirs[i])
			generate_mutated_malware(dirs[i] , model, args)

if __name__ == '__main__':
	main()
	






		

