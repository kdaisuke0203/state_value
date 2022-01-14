import numpy as np
import matplotlib.pyplot as plt
import grid_world

# agentの生成
agent = grid_world.Agent([0,0])

num_row = 7
num_col = 7

# Vの初期化
V = np.zeros((num_row, num_col))
# piの初期化
pi = np.random.randint(0,len(agent.ACTIONS),(num_row,num_col)) # 確定的な方策

print("")
print("pi initial")
print(pi)

print("start iteration")

N = 1000
V_trend = np.zeros((N, num_row, num_col))

count = 0
while(True):
    delta = 0
    for i in range(num_row):
        for j in range(num_col):
            if i == 0 or i == 6 or j == 0 or j == 3 or j ==6:
                tmp = np.zeros(len(agent.ACTIONS))
                v = V[i,j]
                for index,action in enumerate(agent.ACTIONS):
                    #print(index,action)
                #print("delta %f" % delta)
                    agent.set_pos([i,j])
                    s = agent.get_pos()
                    agent.move(action)
                    s_dash = agent.get_pos()
                    tmp[index] =  (agent.reward(s,action) + agent.GAMMA * V[s_dash[0], s_dash[1]])
                V[i,j] = max(tmp)
                V_trend[count, :,:] = V
                delta = max(delta, abs(v - V[i,j]))
    if delta < 1.E-5:
        break
    count += 1

for i in range(num_row):
    for j in range(num_col):
        tmp = np.zeros(len(agent.ACTIONS))
        for index, action in enumerate(agent.ACTIONS):
            agent.set_pos([i,j])
            s = agent.get_pos()
            agent.move(action)
            s_dash = agent.get_pos()
            tmp[index] =  (agent.reward(s,action) + agent.GAMMA * V[s_dash[0], s_dash[1]])
        pi[i,j] = np.argmax(tmp)

V_trend= V_trend[:count,:,:]
       
X, Y = np.mgrid[0:8:1, 0:8:1]
#X-=.0
#Y-=.0
print(V)
z=np.zeros([7,7])
for i in range(7):
    for j in range(7):
        z[i,j] = V[6-j][i]
print(z)
fig, ax = plt.subplots()
im = ax.pcolormesh(X, Y, z, cmap='inferno',vmin=0,vmax=10)
ax.set_xlabel("X")
ax.set_ylabel("Y")
cbar = fig.colorbar(im)
cbar.set_label("Z")
plt.show()
#grid_world.pi_arrow_plot(pi)
grid_world.V_value_plot(V)