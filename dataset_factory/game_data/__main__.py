"""Print two tiny deterministic smoke rows; never writes training data."""
import json

from dataset_factory.game_data import Board, Budget, Nim, decision_row


def main():
    board = Board.initial()
    for action in ((0,), (3,), (1,), (4,)):
        board = board.apply(action)
    states = (board, Nim((1, 3, 5)))
    for index, state in enumerate(states):
        print(json.dumps(decision_row(state, scenario_id=f"smoke-{index}", seed=17,
                                      budget=Budget(2000, 5)), sort_keys=True))


if __name__ == "__main__":
    main()
