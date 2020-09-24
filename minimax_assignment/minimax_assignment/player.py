#!/usr/bin/env python3
import random

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


def shortest_distance(pos1, pos2):
    dist1 = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    dist2 = ((pos1[0] - pos2[0] - 20) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    return min(dist1, dist2)


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate game tree object
        first_msg = self.receiver()
        # Initialize your minimax model
        model = self.initialize_model(initial_data=first_msg)

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(
                model=model, initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def initialize_model(self, initial_data):
        """
        Initialize your minimax model
        :param initial_data: Game data for initializing minimax model
        :type initial_data: dict
        :return: Minimax model
        :rtype: object

        Sample initial data:
        { 'fish0': {'score': 11, 'type': 3},
          'fish1': {'score': 2, 'type': 1},
          ...
          'fish5': {'score': -10, 'type': 4},
          'game_over': False }

        Please note that the number of fishes and their types is not fixed between test cases.
        """
        # EDIT THIS METHOD TO RETURN A MINIMAX MODEL ###
        return None

    def alphabeta(self, node, depth, alpha, beta):
        if depth == 0 or not node.compute_and_get_children():
            fish_pos = node.state.get_fish_positions()
            fish_score = node.state.get_fish_scores()
            hook_pos = node.state.get_hook_positions()
            fish = node.state.get_fish_positions().keys()
            scoreA = sum([fish_score[f] / (0.05 + shortest_distance(hook_pos[0], fish_pos[f])) ** 2 for f in fish])
            scoreB = sum([fish_score[f] / (0.05 + shortest_distance(hook_pos[1], fish_pos[f])) ** 2 for f in fish])
            return scoreA - scoreB

        elif node.state.get_player() == 0:
            v = -1e6
            for child in node.compute_and_get_children():
                v = max(v, self.alphabeta(child, depth - 1, alpha, beta))
                alpha = max(alpha, v)
                if beta <= alpha:
                    break
        else:
            v = 1e6
            for child in node.compute_and_get_children():
                v = min(v, self.alphabeta(child, depth - 1, alpha, beta))
                beta = min(beta, v)
                if beta <= alpha:
                    break
        return v

    def search_best_next_move(self, model, initial_tree_node):
        """
        Use your minimax model to find best possible next move for player 0 (green boat)
        :param model: Minimax model
        :type model: object
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE FROM MINIMAX MODEL ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method!
        child_nodes = initial_tree_node.compute_and_get_children()
        child_v = [self.alphabeta(child, 2, -1e6, 1e6) for child in child_nodes]

        new_pos = (child_nodes[child_v.index(max(child_v))]).state.get_hook_positions()[0]
        curr_pos = initial_tree_node.state.get_hook_positions()[0]
        bestMove = (new_pos[0] - curr_pos[0], new_pos[1] - curr_pos[1])

        MOVE_TO_STR = {
            (0, 0): "stay",
            (0, 1): "up",
            (0, -1): "down",
            (-1, 0): "left",
            (1, 0): "right"
        }
        return MOVE_TO_STR[bestMove]
