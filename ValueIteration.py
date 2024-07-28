import numpy as np, numpy.random as nr
import matplotlib.pyplot as plt
import gymnasium as gym

# Create env
env = gym.make('FrozenLake-v1', render_mode='human')

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

def value_iteration(mdp, gamma, nIt):
    """
    Inputs:
        mdp: MDP
        gamma: discount factor
        nIt: number of iterations, corresponding to n above
    Outputs:
        (value_functions, policies)

    len(value_functions) == nIt+1 and len(policies) == nIt
    """
    print("Iteration | max|V-Vprev| | # chg actions | V[0]")
    print("----------+--------------+---------------+---------")
    Vs = [np.zeros(mdp.nS)] # list of value functions contains the initial value function V^{(0)}, which is zero
    pis = []
    for it in range(nIt):
        oldpi = pis[-1] if len(pis) > 0 else None # \pi^{(it)} = Greedy[V^{(it-1)}]. Just used for printout
        Vprev = Vs[-1] # V^{(it)}

        # Your code should fill in meaningful values for the following two variables
        # pi: greedy policy for Vprev (not V),
        #     corresponding to the math above: \pi^{(it)} = Greedy[V^{(it)}]
        #     ** it needs to be numpy array of ints **
        # V: bellman backup on Vprev
        #     corresponding to the math above: V^{(it+1)} = T[V^{(it)}]
        #     ** numpy array of floats **

        V = np.zeros(mdp.nS) # copy making according to Note 2
        pi = np.zeros(mdp.nS)
        for state in range(mdp.nS):
            possible_vals = []
            for action in range(mdp.nA):
                possible_outcomes = mdp.P[state][action]
                action_val = 0
                for prob, next_state, reward in possible_outcomes:
                    action_val += prob * (reward + gamma * Vprev[next_state])
                possible_vals.append(action_val)
            V[state] = np.max(possible_vals)
            pi[state] = np.argmax(possible_vals)
        diff = np.abs(V - Vprev).max()
        
        nChgActions="N/A" if oldpi is None else (pi != oldpi).sum()
        print("%4i      | %6.5f      | %4s          | %5.3f"%(it, diff, nChgActions, V[0]))
        Vs.append(V)
        pis.append(pi)
        
    plot_state_value_over_iterations(Vs, nIt)

    return Vs, pis

GAMMA = 0.95 # we'll be using this same value in subsequent problems
Vs_VI, pis_VI = value_iteration(mdp, gamma=GAMMA, nIt=20)


for (V, pi) in zip(Vs_VI[:10], pis_VI[:10]):
    plt.figure(figsize=(3,3))
    plt.imshow(V.reshape(4,4), cmap='gray', interpolation='none', clim=(0,1))
    ax = plt.gca()
    ax.set_xticks(np.arange(4)-.5)
    ax.set_yticks(np.arange(4)-.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    Y, X = np.mgrid[0:4, 0:4]
    a2uv = {0: (-1, 0), 1:(0, -1), 2:(1,0), 3:(0, 1)}
    Pi = pi.reshape(4,4)
    for y in range(4):
        for x in range(4):
            a = Pi[y, x]
            u, v = a2uv[a]
            plt.arrow(x, y,u*.3, -v*.3, color='m', head_width=0.1, head_length=0.1)
            plt.text(x, y, str(env.desc[y,x].item().decode()),
                     color='g', size=12,  verticalalignment='center',
                     horizontalalignment='center', fontweight='bold')
    plt.grid(color='b', lw=2, ls='-')
    plt.show()

