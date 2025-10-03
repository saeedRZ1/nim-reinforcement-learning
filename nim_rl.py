import random

class Nim:
    def __init__(self, piles):
        self.piles = piles

    def actions(self):
        return [(i, k) for i, pile in enumerate(self.piles) for k in range(1, pile + 1)]

    def move(self, action):
        pile, count = action
        if self.piles[pile] >= count:
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

    def update_q(self, state, action, reward, future_rewards):
        old_q = self.get_q(state, action)
        self.q[(tuple(state), action)] = old_q + self.alpha * (
            (reward + future_rewards) - old_q
        )

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
        last_move = {1: None, -1: None}
        player = 1
        while not game.is_terminal():
            state = game.piles.copy()
            action = ai.choose_action(state)
            game.move(action)

            if last_move[player]:
                s, a = last_move[player]
                ai.update_q(s, a, 0, ai.best_future_reward(game.piles))

            last_move[player] = (state, action)
            player *= -1

        ai.update_q(*last_move[player], 1, 0)
        ai.update_q(*last_move[-player], -1, 0)
    return ai


def play(ai):
    print("\n🎮 Let's play Nim! Initial piles: [1, 3, 5, 7]")
    game = Nim([1, 3, 5, 7])
    player = 0  # 0 = Human, 1 = AI

    while not game.is_terminal():
        print(f"\nCurrent piles: {game.piles}")
        if player == 0:
            try:
                pile = int(input("Choose pile (0-3): "))
                count = int(input("How many to remove: "))
                action = (pile, count)
                if action not in game.actions():
                    print("❌ Invalid move, try again!")
                    continue
            except ValueError:
                print("❌ Please enter numbers only.")
                continue
        else:
            action = ai.choose_action(game.piles, epsilon=False)
            print(f"🤖 AI chooses: remove {action[1]} from pile {action[0]}")

        game.move(action)
        player = 1 - player

    if player == 0:
        print("🎉 You win!")
    else:
        print("🤖 AI wins!")


if __name__ == "__main__":
    ai = train(3000)
    print("✅ Training complete. AI is ready to play Nim!")
    play(ai)
