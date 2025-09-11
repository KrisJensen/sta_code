import os
import pysta
pysta.reload()
os.chdir(f"{pysta.basedir}")

print("\nRunning all test!")

import test_envs

import test_agents

import test_training

