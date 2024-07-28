import random

class Blackjack:
    def __init__(self):
        self.reset()

    def reset(self):
        self.gambler_sum = self.draw_card() + self.draw_card()
        self.house_sum = self.draw_card() + self.draw_card()
        return self.gambler_sum

    def draw_card(self):
        card = random.randint(2, 11)
        return card

    def gambler_policy(self):
        return self.gambler_sum < 18

    def house_policy(self):
        return self.house_sum <= 15

    def play_gambler(self):
        while self.gambler_policy():
            self.gambler_sum += self.draw_card()
        return self.gambler_sum

    def play_house(self):
        while self.house_policy():
            self.house_sum += self.draw_card()
        return self.house_sum

    def step(self, action):
        if action == "hit":
            self.gambler_sum += self.draw_card()
        done = self.gambler_sum >= 18
        return self.gambler_sum, done

    def play_game(self):
        self.play_gambler()
        self.play_house()
        if self.gambler_sum > 21:
            return -1  # Gambler busts
        elif self.house_sum > 21 or self.gambler_sum > self.house_sum:
            return 1  # Gambler wins
        elif self.gambler_sum == self.house_sum:
            return 0  # Draw
        else:
            return -1  # House wins


class TDZero:
    def __init__(self, alpha=0.1, gamma=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.value_function = {i: 0 for i in range(4, 22)}
        self.value_function[22] = -1  # Bust state

    def update(self, state, reward, next_state):
        next_state = min(next_state, 22) # Cap the next state to 22
        self.value_function[state] += self.alpha * (reward + self.gamma * self.value_function[next_state] - self.value_function[state])
        print(f"This is state: {state} This is next state: {next_state}")
        print(self.value_function)



def simulate_episode(env, policy):
    state = env.reset()
    done = False
    while not done:
        if policy(state):
            state, done = env.step("hit")
        else:
            done = True
    reward = env.play_game()
    return state, reward

class SARSA:
    def __init__(self, alpha=0.1, gamma=1.0, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_values = {(state, action): 0 for state in range(4, 23) for action in ["hit", "stand"]}

    def choose_action(self, state):  # Based on epsilon greedy policy
        if random.random() < self.epsilon:
            return random.choice(["hit", "stand"])
        else:
            hit_value = self.q_values[(state, "hit")]
            stand_value = self.q_values[(state, "stand")]
            return "hit" if hit_value > stand_value else "stand" # for specific state, we see what action has max value in Q table

    def update(self, state, action, reward, next_state, next_action): # Updates Q table
        current_q = self.q_values[(state, action)]
        next_q = self.q_values[(next_state, next_action)]
        self.q_values[(state, action)] += self.alpha * (reward + self.gamma * next_q - current_q)

    def extract_policy(self): # Find the optimal policy at the end
        policy = {}
        for state in range(4, 23):
            hit_value = self.q_values[(state, "hit")]
            stand_value = self.q_values[(state, "stand")]
            policy[state] = "hit" if hit_value > stand_value else "stand"
        return policy

    def state_value_function(self):
        value_function = {state: max(self.q_values[(state, "hit")], self.q_values[(state, "stand")]) for state in range(4, 23)}
        return value_function

def gambler_policy(state):
    return state < 18

def main_sarsa():
    env = Blackjack()
    sarsa = SARSA(alpha=0.1, gamma=1.0, epsilon=0.1)
    num_episodes = 10000

    for episode in range(num_episodes):
        state = env.reset()
        action = sarsa.choose_action(state)
        done = False

        while not done:
            next_state, done = env.step(action)
            if next_state > 21:
                next_state = 22
                reward = -1
                next_action = "stand"
                done = True
            else:
                reward = 0 if not done else env.play_game()
                next_action = sarsa.choose_action(next_state)

            sarsa.update(state, action, reward, next_state, next_action)
            state, action = next_state, next_action

    # Extract the optimal policy and state value function
    optimal_policy = sarsa.extract_policy()
    state_value_function = sarsa.state_value_function()

    # Print the optimal policy and state value function
    print("Optimal policy:")
    print(state_value_function)
    state_value_function[22] = 0
    for i in range(4,22):
        state_value_function[i] = abs(state_value_function[i])
    for i in range(4,22):
        state_value_function[i] = state_value_function[i] / max(state_value_function.values())

    for state in sorted(optimal_policy.keys()):
        if state == 22:
            print(f"State {state}: automatic loose, Estimated probability of winning for the gambler: {state_value_function[state]}")
        else:
            print(f"State {state}: {optimal_policy[state]} , Estimated probability of winning for the gambler: {state_value_function[state]}")


def main_td0():
    env = Blackjack()
    td = TDZero(alpha=0.1, gamma=1.0)
    num_episodes = 10
    reset = 0

    for _ in range(num_episodes):
        if reset == 0:
            state = env.reset()
        done = False
        reset = 0
        while not done:
            action = "hit" if gambler_policy(state) else "stand"
            next_state, done = env.step(action)
            if next_state > 21:
                next_state = env.reset()
                td.update(state, -1, next_state)
                reset = 1
                done = True
                state = next_state
            else:
                reward = 0 if not done else env.play_game()
                td.update(state, reward, next_state)
                state = next_state


    # Calculate the probabilities of winning from every initial state
    td.value_function[22] = 0
    print(td.value_function)
    maxValue = max(td.value_function.values())
    for i in range(4,22):
        td.value_function[i] = td.value_function[i] / maxValue
    win_probabilities = (td.value_function)
    for state in win_probabilities.keys():
        print(f"State {state}, Estimated probability of winning for the gambler: {win_probabilities[state]}")


if __name__ == "__main__":
    print("################# SECTION 1 - TD(0) ###################")
    print("\n")
    main_td0()
    print("\n")
    print("################# SECTION 2 - SARSA ###################")
    print("\n")
    main_sarsa()