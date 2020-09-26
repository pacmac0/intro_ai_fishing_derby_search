#!/usr/bin/env python3
import random
import sys

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


global best_move
best_move = 0

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
        first_msg = self.receiver() # initial fish in the scenario
        # Initialize your minimax model use model as a memory
        model = self.initialize_model(initial_data=first_msg)

        while True:
            msg = self.receiver()
            """
            {'game_over': False, 'hooks_positions': {0: (6, 8), 1: (9, 17)}, 'fishes_positions': {0: (6, 17), 4: (1, 9)}, ... , 'fish_scores': {0: 11, 4: 11}, 'player_scores': {0: 0, 1: 14}, 'caught_fish': {0: None, 1: None}}
            """

            # Create the root node of the game tree
            node = Node(message=msg, player=0)
            # TODO update model with new nodes

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(model=model, initial_tree_node=node)

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

    def  minimaxAlphabeta(self, node, depth, alpha, beta, player):
        if (depth == 0) or (len(node.compute_and_get_children())) == 0: # terminal state
            v = node.state.player_scores[0] - node.state.player_scores[1] # TODO use a heuristic to score the state
        elif player == 0: # no terminal state
            v = {'eval':-float('inf'),
                 'move':0}
            for child in node.children:
                v = max(v, self.minimaxAlphabeta(child, depth-1, alpha, beta, 1))
                alpha = max(alpha, v)
                if beta <= alpha:
                    break # beta prune
                best_move = child.move
            
        else: # player == 1
            v = {'eval':float('inf'),
                 'move':0}
            for child in node.children:
                v = min(v, self.minimaxAlphabeta(child, depth-1, alpha, beta, 0))
                beta = min(beta, v)
                if beta <= alpha:
                    break # alpha prune
                best_move = child.move
        print(v)
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


        _ = self.minimaxAlphabeta(initial_tree_node, 
                            3, # depth
                            initial_tree_node.state.player_scores[0] - initial_tree_node.state.player_scores[1], # alpha
                            initial_tree_node.state.player_scores[1] - initial_tree_node.state.player_scores[0], # beta
                            initial_tree_node.player)
        
        print(best_move)
        print()
        
        #return ACTION_TO_STR[move[0]]


        random_move = random.randrange(5)
        return ACTION_TO_STR[random_move]