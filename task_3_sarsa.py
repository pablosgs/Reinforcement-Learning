# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 09:55:50 2021

@author: Pablo
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
# %pylab inline
import random
import numpy as np
import time


# parameters
gamma = 0.8 # discounting rate
rewardSize = -1
gridSize = 9
alpha = 0.5 # (0,1] // stepSize
terminationStates = [[6,5], [gridSize-1, gridSize-1]]
actions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
numIterations = 10000
epsilon = 0.5
# initialization
Q_u = np.zeros((gridSize, gridSize))
Q_d = np.zeros((gridSize, gridSize))
Q_r = np.zeros((gridSize, gridSize))
Q_l = np.zeros((gridSize, gridSize))
returns = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
deltas = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]
# utils
def generateInitialState():
    initState = random.choice(states[1:-1])
    while -1 in list(initState) or gridSize in list(initState) or list(initState)==[7,1] or list(initState)==[7,2] or list(initState)==[7,3] or list(initState)==[7,4] or list(initState)==[1,2] or list(initState)==[1,3] or list(initState)==[1,4] or list(initState)==[1,5] or list(initState)==[1,6] or list(initState)==[2,6] or list(initState)==[3,6] or list(initState)==[4,6] or list(initState)==[5,6]:
        initState = random.choice(states[1:-1])
    return initState

def generateNextAction(state, Q_u, Q_d, Q_r, Q_l, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    else:
        q=[]
        for i in actions:
            new_state = np.array(state)+np.array(i)
            if new_state[0]<8 and new_state[0]>0 and new_state[1]<8 and new_state[1]>0:
                if i == [1,0]:
                    q.append(Q_u[new_state[0],new_state[1]])
                elif i == [-1,0]:
                    q.append(Q_d[new_state[0],new_state[1]])
                elif i == [0,1]:
                    q.append(Q_r[new_state[0],new_state[1]])
                elif i == [0,-1]:
                    q.append(Q_l[new_state[0],new_state[1]])
            else:
                q.append(-1000)                
    return actions[np.argmax(q)]


def takeAction(state, action):
    finalState = np.array(state)+np.array(action)
    if -1 in list(finalState) or gridSize in list(finalState) or list(finalState)==[7,1] or list(finalState)==[7,2] or list(finalState)==[7,3] or list(finalState)==[7,4] or list(finalState)==[1,2] or list(finalState)==[1,3] or list(finalState)==[1,4] or list(finalState)==[1,5] or list(finalState)==[1,6] or list(finalState)==[2,6] or list(finalState)==[3,6] or list(finalState)==[4,6] or list(finalState)==[5,6]:
        reward = 0
        finalState = state
        
    elif list(state) == [gridSize-1, gridSize-1]:
        finalState = state
        reward = 50

        
    elif list(state) == [6,5]:
        finalState = state
        reward = -50

    else:
        reward = rewardSize

    return reward, finalState

start = time.time()
for it in tqdm(range(numIterations)):
    state = generateInitialState()
    while True:
        action = generateNextAction(state, Q_u, Q_d, Q_r, Q_l, epsilon)
        reward, finalState = takeAction(state, action)
        
        # we reached the end
        if list(finalState) == [gridSize-1, gridSize-1] or list(finalState) == [6,5]:
            break
        next_action = generateNextAction(finalState, Q_u, Q_d, Q_r, Q_l, epsilon)
        
        # modify Value function
        if next_action == [1,0]:
            before =  Q_u[state[0], state[1]]
            Q_u[state[0], state[1]] += alpha*(reward + gamma*Q_u[finalState[0], finalState[1]] - Q_u[state[0], state[1]])
            deltas[state[0], state[1]].append(float(np.abs(before-Q_u[state[0], state[1]])))
        
        if next_action == [-1,0]:
            before =  Q_d[state[0], state[1]]
            Q_d[state[0], state[1]] += alpha*(reward + gamma*Q_d[finalState[0], finalState[1]] - Q_d[state[0], state[1]])
            deltas[state[0], state[1]].append(float(np.abs(before-Q_d[state[0], state[1]])))
        
        if next_action == [0,1]:
            before =  Q_r[state[0], state[1]]
            Q_r[state[0], state[1]] += alpha*(reward + gamma*Q_r[finalState[0], finalState[1]] - Q_r[state[0], state[1]])
            deltas[state[0], state[1]].append(float(np.abs(before-Q_r[state[0], state[1]])))
        
        if next_action == [0,-1]:
            before =  Q_l[state[0], state[1]]
            Q_l[state[0], state[1]] += alpha*(reward + gamma*Q_l[finalState[0], finalState[1]] - Q_l[state[0], state[1]])
            deltas[state[0], state[1]].append(float(np.abs(before-Q_l[state[0], state[1]])))
        
        state = finalState
end=time.time()
tot_time=end-start

final_Q = []  
for i in range(9):
    for j in range(9):
        q = []
        q.append(Q_u[i,j])
        q.append(Q_d[i,j])
        q.append(Q_r[i,j])
        q.append(Q_l[i,j])
        maxi = np.argmax(q)
        final_Q.append(maxi)
new_Q = []
new_q = []


for i in range(9):
    new_q.append(final_Q[i*9:(i+1)*9])



X=[]
Y=[]
U=[]
V=[]

for i in range(9):
    for j in range(9):
        if [i,j] == [7,1] or [i,j] == [7,2] or [i,j] == [7,3] or [i,j] == [7,4] or [i,j] == [1,2] or [i,j] == [1,3] or [i,j] == [1,4] or [i,j] == [1,5] or [i,j] == [1,6] or [i,j] == [2,6] or [i,j] == [3,6] or [i,j] == [4,6] or [i,j] == [5,6] or [i,j] == [6,5] or [i,j] == [8,8]:
            wall = True
        else:
            wall = False
        if wall == False:
            X.append(j)
            Y.append(i)
            if new_q[i][j] == 0:
                U.append(0)
                V.append(1)
            if new_q[i][j] == 1:
                U.append(0)
                V.append(-1)
            if new_q[i][j] == 2:
                U.append(1)
                V.append(0)
            if new_q[i][j] == 3:
                U.append(-1)
                V.append(0)
            
for i in range(len(X)):
    X[i] += 0.5
for i in range(len(Y)):
    Y[i] += 0.5
fig, ax = plt.subplots()         
q = ax.quiver(X, Y, U,V)
plt.gca().invert_yaxis()
