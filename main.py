import random
import numpy as np

class Nim:
    def __init__(self, piles):
        self.piles = piles

    def actions(self):
        return [(i, k) for i, pile in enumerate(self.piles) for k in range(1, pile+1)]

    def move(self, action):
        pile, count = action
        self.piles[pile] -= count

    def is_terminal(self):
        return all(p == 0 for p in self.piles)

class NimAI:
    def __init__(self, alpha=0.5, epsilon=0.1):
        self.q = {}
        self.alpha = alpha
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.q.get((tuple(state), action), 0)

    def update_q(self, state, action, old_q, reward, future_rewards):
        self.q[(tuple(state), action)] = old_q + self.alpha * ((reward + future_rewards) - old_q)

    def best_future_reward(self, state):
        actions = Nim(state).actions()
        if not actions:
            return 0
        return max([self.get_q(state, a) for a in actions])

    def choose_action(self, state, epsilon=True):
        actions = Nim(state).actions()
        if not actions:
            return None
        if epsilon and random.random() < self.epsilon:
            return random.choice(actions)
        q_values = [self.get_q(state, a) for a in actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(best_actions)

def train(n=1000):
    ai = NimAI()
    for _ in range(n):
        game = Nim([1, 3, 5, 7])
        last = {1: None, -1: None}
        player = 1
        while not game.is_terminal():
            state = game.piles.copy()
            action = ai.choose_action(state)
            game.move(action)
            if last[player]:
                ai.update_q(*last[player], 0, ai.best_future_reward(game.piles))
            last[player] = (state, action, ai.get_q(state, action), None)
            player *= -1
        ai.update_q(*last[player], 1, 0)
        ai.update_q(*last[-player], -1, 0)
    return ai

if __name__ == "__main__":
    ai = train(3000)
    print("Training complete. AI is ready to play Nim!")
