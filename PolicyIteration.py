##################################
import numpy as np, numpy.random as nr
import matplotlib.pyplot as plt

# Create env
import gymnasium as gym
env = gym.make('FrozenLake-v1')
env = env.env
print(env.__doc__)
print("")

#################################
# Some basic imports and setup
# Let's look at what a random episode looks like.

import matplotlib.pyplot as plt
#%matplotlib inline
np.set_printoptions(precision=3)

# Seed RNGs so you get the same printouts as me
#env.seed(0); from gymnasium.spaces import prng; prng.seed(10)
# Generate the episode
env.reset(seed=10) # I added seed=0 as gymnasium seed instead of this line
# for t in range(100):
#     env.render()
#     a = env.action_space.sample()
#     ob, rew, done, _ = env.step(a)
#     if done:
#         break
# assert done
# env.render();

#################################
# Create MDP for our env
# We extract the relevant information from the gym Env into the MDP class below.
# The `env` object won't be used any further, we'll just use the `mdp` object.

class MDP(object):
    def __init__(self, P, nS, nA, desc=None):
        self.P = P # state transition and reward probabilities, explained below
        self.nS = nS # number of states
        self.nA = nA # number of actions
        self.desc = desc # 2D array specifying what each grid cell means (used for plotting)
mdp = MDP( {s : {a : [tup[:3] for tup in tups] for (a, tups) in a2d.items()} for (s, a2d) in env.P.items()}, 16, 4, env.desc)
GAMMA = 0.95 # we'll be using this same value in subsequent problems

print("")
print("mdp.P is a two-level dict where the first key is the state and the second key is the action.")
print("The 2D grid cells are associated with indices [0, 1, 2, ..., 15] from left to right and top to down, as in")
print(np.arange(16).reshape(4,4))
print("Action indices [0, 1, 2, 3] correspond to West, South, East and North.")
print("mdp.P[state][action] is a list of tuples (probability, nextstate, reward).\n")
print("For example, state 0 is the initial state, and the transition information for s=0, a=0 is \nP[0][0] =", mdp.P[0][0], "\n")
print("As another example, state 5 corresponds to a hole in the ice, in which all actions lead to the same state with probability 1 and reward 0.")
for i in range(4):
    print("P[5][%i] =" % i, mdp.P[5][i])
print("")

#################################
# Programing Question No. 2, part 1 - implement where required.

def compute_vpi(pi, mdp, gamma):
    # use pi[state] to access the action that's prescribed by this policy
    V = np.ones(mdp.nS)
    b = np.zeros(mdp.nS)
    A = np.zeros(shape=(mdp.nS, mdp.nS))
    for state in range(mdp.nS):
        action = pi[state]
        possible_outcomes = mdp.P[state][action]
        for prob, next_state, reward in possible_outcomes:
            b[state] += prob * reward
            A[state][next_state] += prob * gamma
    A=np.identity(mdp.nS)-A
    V = np.linalg.solve(A,b)
    return V


actual_val = compute_vpi(np.arange(16) % mdp.nA, mdp, gamma=GAMMA)
print("Policy Value: ", actual_val)

#################################
# Programing Question No. 2, part 2 - implement where required.

def compute_qpi(vpi, mdp, gamma):
    Qpi = np.zeros([mdp.nS, mdp.nA])
    for state in range(mdp.nS):
        for action in range(mdp.nA):
            possible_outcomes = mdp.P[state][action]
            for prob, next_state, reward in possible_outcomes:
                Qpi[state][action] += prob * (reward + gamma * vpi[next_state])
    return Qpi

Qpi = compute_qpi(np.arange(mdp.nS), mdp, gamma=0.95)
print("Policy Action Value: ", actual_val)

#################################
# Programing Question No. 2, part 3 - implement where required.
# Policy iteration

def policy_iteration(mdp, gamma, nIt):
    Vs = []
    pis = []
    pi_prev = np.zeros(mdp.nS,dtype='int')
    pis.append(pi_prev)
    print("Iteration | # chg actions | V[0]")
    print("----------+---------------+---------")
    for it in range(nIt):
        # you need to compute qpi which is the state-action values for current pi
        vpi = compute_vpi(pi_prev, mdp, gamma)
        qpi = compute_qpi(vpi, mdp, gamma)
        pi = qpi.argmax(axis=1)
        print("%4i      | %6i        | %6.5f"%(it, (pi != pi_prev).sum(), vpi[0]))
        Vs.append(vpi)
        pis.append(pi)
        pi_prev = pi
    
    #plotting of the values per state
    plot_state_value_over_iterations(Vs, nIt)
    
    return Vs, pis


def plot_state_value_over_iterations(state_values, iteration_count):
    # Create a list of iteration numbers
    iteration_numbers = [i for i in range(1, iteration_count + 1)]
    
    # Set the size of the figure
    plt.figure(figsize=(8, 5))
    
    # For each state, plot the state value over iterations
    for state in range(mdp.nS):
        # Get the state value for each iteration
        state_value = [state_values[it][state] for it in range(iteration_count)]
        # Plot the state value over iterations
        plt.plot(iteration_numbers, state_value, label=f'state {state}')
    
    # Set the labels and title of the plot
    plt.xlabel('Iterations')
    plt.ylabel('State value')
    plt.title(f'Value function per state over {iteration_count} iterations')
    
    # Set the x-ticks to be the iteration numbers
    plt.xticks(iteration_numbers)
    # Add a legend to the plot at the top right corner
    plt.legend(bbox_to_anchor=(1.15, 1), loc=1)
    # Add a grid to the plot
    plt.grid()
    plt.show()

nIt=20
Vs_PI, pis_PI = policy_iteration(mdp, gamma=0.95, nIt=20)




