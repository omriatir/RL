import random
import gymnasium as gym
import numpy as np

# Initialize the Blackjack environment from Gym
env = gym.make('Blackjack-v1', natural=True)

def gambler_policy(state):
    # Policy: If the sum is 18 or more, then stand; otherwise, hit
    return 0 if sum(state[0]) >= 18 else 1

def td0(num_episodes=10000, alpha=0.1, gamma=1.0):
    # Initialize the state values
    V = np.zeros(32)  # State values for states 0 to 31
    state_count = np.zeros(32)  # To count occurrences of each state

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        print(state)
        while not done:
            action = gambler_policy(state)

            next_state, reward, done, _, _ = env.step(action)
            print(next_state)

            # TD(0) update
            state_value = V[state[0][0]]
            next_state_value = V[next_state[0][0]]
            V[state[0][0]] += alpha * (reward + gamma * next_state_value - state_value)

            # Track state occurrences
            state_count[state[0][0]] += 1

            state = next_state

    return V, state_count


def estimate_probability_of_winning(num_episodes=10000):
    V, state_count = td0(num_episodes)

    # Calculate the estimated probability of winning
    win_count = 0
    total_states = 0

    for state in range(12, 22):  # States where gambler would stand
        if state_count[state] > 0:
            win_count += V[state]  # Aggregate state values as win counts
            total_states += state_count[state]

    probability_of_winning = win_count / total_states if total_states > 0 else 0
    return probability_of_winning


if __name__ == "__main__":
    print("Estimated probability of winning:", estimate_probability_of_winning())