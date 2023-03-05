SNOWCAP_RUNS = 2


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from BinaryTreeCustom import *
import random
import math

root = tree(9, is_perfect=True)
for node in root.leaves:
    node.left = tree(2, is_perfect=True)
    node.right = tree(2, is_perfect=True)

root.address = ''
root.n = 0
root.t = 0

for node in root:
    node.value = 0
    node.t = 0
    
iter = 1
for i in range((root.height)+ 1):
    if i == 0:
        continue
    level = root.levels[i]
    for node in level:
        parent_node = get_parent(root = root, child = node)
        parent_address = parent_node.address
        node.n = 0

        if iter % 2 == 0:
            node.address = parent_address + 'R'
            iter = 1
           
        else: 
            node.address = parent_address + 'L'
            iter += 1
           

level = root.levels[root.height]
for node in level:
    parent_node = get_parent(root = root, child = node)
    parent_address = parent_node.address
    node.address = parent_address

iter = 1

for node in root.leaves:
    if iter == 1:
        node.address += 'L'
        iter += 1
    else:
        node.address += 'R'
        iter -= 1


random_leaf = random.randint(0, len(root.leaves) - 1)
random_address = root.leaves[random_leaf].address

for node in root.leaves:
    current_leaf_address = node.address
    edit_distance = 0
    for i in range(len(node.address)):
        if node.address[i] != random_address[i]:
            edit_distance += 1
    
    node.value = round((1000 * math.e ** (-edit_distance/2)), 2)
    print('leaf:',node.address, 'distance:', edit_distance)
    


def rollout(node):
    new_node = node
    original_node = node
    value_list = []
    for i in range(SNOWCAP_RUNS):
        backup_node = original_node
        new_node = original_node
        a = True
        while a == True:
            j = random.randint(0, 10)
            
            if i == 0:
                backup_node = original_node
            else:
                backup_node = new_node
             
            if j % 2 == 0:
                try:
                    new_node = new_node.left
                    address = new_node.address
                    value = new_node.value
                except:
                    value_list.append(backup_node.value)
                    a = False
            else:
                try:
                    new_node = new_node.right
                    address = new_node.address
                    value = new_node.value
                except:
                    value_list.append(backup_node.value)
                    a = False
                    
    return sum(value_list)/SNOWCAP_RUNS



def reset(root):
    for node in root:
        node.t = 0
        node.n = 0
        node.value = 0
    
    for node in root.leaves:
        current_leaf_address = node.address
        edit_distance = 0
        for i in range(len(node.address)):
            if node.address[i] != random_address[i]:
                edit_distance += 1
        # node.value = round((1000 * math.e ** (-edit_distance/5)), 2)
        node.value = round((1000 * math.e ** (-edit_distance/2)))
        
    return root


def update_values(child_node):
    node = child_node
    node_value = rollout(node)
    print(node_value)
    node.t += node_value
    node.n += 1

    while True:
        try:
            parent = get_parent(root, node)
            parent.t += node_value
            parent.n += 1
            node = parent
        except:
            break

    
def traversal(root, UCB):
    new_nodes = 0
    node = root
    new_node = node
    val = -1
    while val != 0:
        try:
            new_node = node
            if node.left.t == 0:
                new_node = new_node.left
                new_nodes += 1
                # print('check1')
            
            elif node.right.t == 0:  
                new_node = new_node.right
                new_nodes += 1
                # print('check2')
            
            else:
                left_ucb = int((node.left.t/node.left.n) + UCB * math.sqrt((math.log(get_parent(root, node.left).n))/node.left.n))
                right_ucb = int((node.right.t/node.right.n) + UCB * math.sqrt((math.log(get_parent(root, node.right).n))/node.right.n))
                # print('check3')
                if np.argmax([left_ucb, right_ucb]) == 0:
                    new_node = new_node.left
                    # print('check4')
                if np.argmax([left_ucb, right_ucb]) == 1:
                    new_node = new_node.right
                    # print('check5')
            node = new_node
            val = new_node.t
        except:
            # print('check6')
            break

    return new_node, new_nodes
    
def mcts(root, UCB):
    import time
    checked_nodes = 0
    count = 0
    val = 0
    start = time.time()
    while val != 1000:
        count += 1
        trav_node, new_nodes = traversal(root, UCB)
        checked_nodes += new_nodes
        print(trav_node.value)
        final = trav_node.value
        update_values(trav_node)
        val = trav_node.value
        if count == 50:
            break
    end = time.time()
    time = end - start
    return count, time, checked_nodes


def main(root):
    UCB = 0
    final_count = np.zeros((10,10))
    final_time = np.zeros((10,10))
    final_check = np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            count, time, checked_nodes = mcts(root, UCB)
            final_count[i,j] = count
            final_time[i,j] = time
            final_check[i,j] = checked_nodes
            root = reset(root)
        UCB += 0.5
    return final_count, final_time, final_check
            
            
# mean_count = []          
# for i in range(10):
#     mean_count.append(np.mean(final_count[i]))

# c = []
# ucb=0
# for i in range(10):
#     c.append(ucb)
#     ucb += 0.5
    
# pyplot.plot(c,f_count)
# pyplot.xlabel('c')
# pyplot.ylabel('Mean number of iterations')
# pyplot.title('Number of iterations needed to find the optimal value depending on c')



    
