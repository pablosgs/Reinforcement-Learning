# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 12:39:05 2021

@author: Pablo
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
# %pylab inline
import random

# parameters
gamma = 0.9 # discounting rate
rewardSize = -1
gridSize = 9
terminationStates = [[6,5], [gridSize-1, gridSize-1]]
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
numIterations = 10000
# initialization
V = np.zeros((gridSize, gridSize))
returns = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
deltas = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]
# utils
def generateEpisode():
    initState = random.choice(states[1:-1])
    episode = []
    while True:
        if list(initState) in terminationStates:
            return episode
        action = random.choice(actions)
        finalState = np.array(initState)+np.array(action)
        if -1 in list(finalState) or gridSize in list(finalState) or list(finalState)==[7,1] or list(finalState)==[7,2] or list(finalState)==[7,3] or list(finalState)==[7,4] or list(finalState)==[1,2] or list(finalState)==[1,3] or list(finalState)==[1,4] or list(finalState)==[1,5] or list(finalState)==[1,6] or list(finalState)==[2,6] or list(finalState)==[3,6] or list(finalState)==[4,6] or list(finalState)==[5,6]:
            finalState = initState
        if list(finalState) == [gridSize-1, gridSize-1]:
            episode.append([list(initState), action, 50, list(finalState)])
        elif list(finalState) == [6,5]:
            episode.append([list(initState), action, -50, list(finalState)])
        else:
            episode.append([list(initState), action, rewardSize, list(finalState)])
        initState = finalState
for it in tqdm(range(numIterations)):
    episode = generateEpisode()
    G = 0
    #print(episode)
    for i, step in enumerate(episode[::-1]):
        G = gamma*G + step[2]
        if step[0] not in [x[0] for x in episode[::-1][len(episode)-i:]]:
            idx = (step[0][0], step[0][1])
            returns[idx].append(G)
            newValue = np.average(returns[idx])
            deltas[idx[0], idx[1]].append(np.abs(V[idx[0], idx[1]]-newValue))
            V[idx[0], idx[1]] = newValue


           
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()
uniform_data = V
ax = sns.heatmap(uniform_data, cmap = 'viridis')


