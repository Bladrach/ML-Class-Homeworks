from qmap_TM import QMap
import numpy as np
import time
import os
import pylab as plt
import xlsxwriter
# create quest map
qmap = QMap()

# QTable : contains the Q-Values for every (state,action) pair
qtable = np.random.rand(qmap.stateCount, qmap.actionCount).tolist()

# hyperparameters
epochs = 50
alpha = 0.8
gamma = 0.99
epsilon = 0.1
decay = 0.1
total_rewards = []
total_reward = 0
# training loop
for i in range(epochs):
    state, reward, done, TM = qmap.reset()
    steps = 0
    bounty = 0
    TM = 0
    scores = []
    bounties = []  
    # act randomly sometimes to allow exploration
    if np.random.uniform() < epsilon:
        action = qmap.randomAction()
    # if not select max action in Qtable (act greedy)
    else:
        action = qtable[state].index(max(qtable[state]))
    while not done:
        os.system('cls')
        print("epoch #", i+1, "/", epochs)
        qmap.render()
        time.sleep(0.05)

        # count steps to finish game
        steps += 1
        next_state = action
        # act randomly sometimes to allow exploration
        if np.random.uniform() < epsilon:
            next_action = qmap.randomAction()
        # if not select max action in Qtable (act greedy)
        else:
            next_action = qtable[next_state].index(max(qtable[next_state]))

        # take action
        if steps == 100 and TM == 1:
            next_state, reward, done, TM = qmap.step(action)
            next_action = qtable[next_state].index(max(qtable[next_state])) 
            reward = -200
            bounty += reward
            total_reward += reward
            bounties.append(bounty)
            total_rewards.append(total_reward)
            TM = 1
            break
        elif steps == 100:
            next_state, reward, done, TM = qmap.step(action)
            next_action = qtable[next_state].index(max(qtable[next_state])) 
            reward = -100
            bounty += reward
            total_reward += reward
            bounties.append(bounty)
            total_rewards.append(total_reward)
            break
        elif TM == 1:
            next_state, reward, done, TM = qmap.step(action)
            next_action = qtable[next_state].index(max(qtable[next_state])) 
            bounty += reward
            total_reward += reward
            bounties.append(bounty)
            total_rewards.append(total_reward)
            TM = 1
            break

        else:
            next_state, reward, done, TM = qmap.step(action)
            next_action = qtable[next_state].index(max(qtable[next_state])) 

        # update qtable value with Bellman equation
        qtable[state][action] += alpha * (reward + gamma * qtable[next_state][next_action] - qtable[state][action])

        # update state
        state = next_state
        action = next_action
        scores.append(state)
        bounty += reward
        total_reward += reward
        bounties.append(bounty)
        total_rewards.append(total_reward)
    # The more we learn, the less we take random actions
    epsilon -= decay*epsilon

    if steps == 100 and TM == 1:
        print("\nWalked so long and killed by toxic mist  in", steps, "steps...".format(steps))
        print("\nReward = ", reward, " gold...")
        plt.figure(1)
        plt.subplot(121)
        plt.plot(scores)
        plt.title("Followed path for this epoch (Get lost and killed by toxic mist)")
        plt.xlabel('Steps')
        plt.ylabel('States for corresponding steps')
        plt.grid(True)
    
        plt.subplot(122)
        plt.plot(bounties)
        plt.title("Taken rewards for each step in this epoch")
        plt.xlabel('Steps')
        plt.ylabel('Taken rewards for corresponding steps')
        plt.grid(True)
        plt.show()
        time.sleep(1.2)

    elif steps == 100:
        print("\nFailed to find basilisk in", steps, "steps...".format(steps))
        print("\nReward = ", reward, " gold...")
        plt.figure(1)
        plt.subplot(121)
        plt.plot(scores)
        plt.title("Followed path for this epoch (Failed)")
        plt.xlabel('Steps')
        plt.ylabel('States for corresponding steps')
        plt.grid(True)
    
        plt.subplot(122)
        plt.plot(bounties)
        plt.title("Taken rewards for each step in this epoch")
        plt.xlabel('Steps')
        plt.ylabel('Taken rewards for corresponding steps')
        plt.grid(True)
        plt.show()
        time.sleep(1.2)

    elif TM == 1:
        print("\nKilled by toxic mist in", steps, "steps...".format(steps))
        print("\nReward = -100 gold...")
        plt.figure(1)
        plt.subplot(121)
        plt.plot(scores)
        plt.title("Followed path for this epoch (Killed by toxic mist)")
        plt.xlabel('Steps')
        plt.ylabel('States for corresponding steps')
        plt.grid(True)
    
        plt.subplot(122)
        plt.plot(bounties)
        plt.title("Taken rewards for each step in this epoch")
        plt.xlabel('Steps')
        plt.ylabel('Taken rewards for corresponding steps')
        plt.grid(True)
        plt.show()
        time.sleep(1.2)

    else:
        print("\nBasilisk has been successfully slained in", steps, "steps!!".format(steps))
        print("\nReward = ", reward, " gold!!")
        plt.figure(1)
        plt.subplot(121)    
        plt.plot(scores)
        plt.title("Victory road for this epoch (Successed)")
        plt.xlabel('Steps')
        plt.ylabel('States for corresponding steps')
        plt.grid(True)

        plt.subplot(122)
        plt.plot(bounties)
        plt.title("Taken rewards for each step in this epoch")
        plt.xlabel('Steps')
        plt.ylabel('Taken rewards for corresponding steps')
        plt.grid(True)
        plt.show()
        time.sleep(1.2)

plt.plot(total_rewards)
plt.title("Total reward for all epochs")
plt.xlabel('Total steps')
plt.ylabel('Total reward')
plt.grid(True)
plt.show()

QTABLE_excel = xlsxwriter.Workbook('Qtable_SARSA_TM.xlsx')
worksheet = QTABLE_excel.add_worksheet()
worksheet.write(0, 0, "[Actions, States]")
worksheet.write(1, 0, "Left")
worksheet.write(2, 0, "Right")
worksheet.write(3, 0, "Up")
worksheet.write(4, 0, "Down")

for col, data in enumerate(qtable):
    row = 1
    worksheet.write_column(row, col+1, data)

i = 0
row = 0
col = 1
while i < 100:
    worksheet.write_number(row, col, i)
    i += 1
    col += 1

QTABLE_excel.close()
