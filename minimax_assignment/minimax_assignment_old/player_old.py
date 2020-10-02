#!/usr/bin/env python3
import random
import sys

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR

import time
import math

# global variables for RSC (repetive state checking)
fish_to_type = {}
zobristTable = []
lookUpTable = dict()

start_time = 0
time_limit = 40 # 2 digits!
max_score = 0
time_out = False

def shortest_distance_squared(pos1, pos2):
    return min((pos1[0] - pos2[0]) ** 2, (pos1[0] - pos2[0] - 20) ** 2) + (pos1[1] - pos2[1]) ** 2

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
            global lookUpTable
            if any(c is not None for c in node.state.get_caught()) and len(lookUpTable) > 0:
                lookUpTable = dict()
            
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
        # init the hash table for zobrist hashing

        # get number different fish type
        del initial_data['game_over']
        types = [initial_data[fish]['type'] for fish in initial_data]
        types = set(types)
        
        # max score achieveable in game
        global max_score
        max_score = sum([initial_data[f]['score'] for f in initial_data if initial_data[f]['score'] > 0])

        global fish_to_type
        fish_to_type = {''.join(c for c in k if c.isdigit()):v['type'] for (k,v) in initial_data.items()}

        d = {k: v for k, v in zip(types, [0 for _ in range(len(types))])}
        d.update(dict(h1=0))
        d.update(dict(h2=0))
        
        # init the zobrist table
        global zobristTable
        zobristTable = [[d.copy() for _ in range(20)] for _ in range(20)]
        
        for i in range(len(zobristTable)): # x position
            for j in range(len(zobristTable[0])):  # y position
                for key in d.keys():  # fish type + hooks
                    zobristTable[i][j][key] = random.randint(0, 1000000000)

        return None

    def alphabeta(self, node, depth, alpha, beta):
        
        state = node.state
        children = node.compute_and_get_children()
        # move ordering, to evaluate the states with the same move as the previous state first
        children.sort(key=lambda x: (x.move - node.move) ** 2, reverse=False)
        # we could delete the child with illigal move if the enemy is next to us and we cant move that way

        # if max depth reached, end of game (terminate state) reached, time was just tried out
        if (time.time() - start_time) >= time_limit*pow(10,-3):
            global time_out
            time_out = True
        if time_out or depth == 0 or (not children):
            
            fish_pos = state.get_fish_positions()
            fish_score = state.get_fish_scores()
            hook_pos = state.get_hook_positions()
            scores = state.get_player_scores()
            caught = state.get_caught()

            # RSC
            
            
            # print(fish_pos) # {0: (7, 19), 1: (5, 15), 4: (19, 6)} with key = fish_number
            # print(hook_pos) # {0: (8, 19), 1: (9, 16)}
            # print(fish_to_type) # {'0': 11, '1': 1, '2': 10, '3': 1, '4': 11}
                                  
            # for fish_id, coordinates in fish_pos.items():
            #     print("fish {} is at {},{} ad has type {}".format(fish_id, coordinates[0], coordinates[1], fish_to_type[str(fish_id)]))
            
            # calculate key-value of state
            key = 0
            for fish_id, coordinates in fish_pos.items():
                # get fish-type to look up zobrist value
                key ^= zobristTable[ coordinates[0] ][ coordinates[1] ][ fish_to_type[str(fish_id)] ]
            # XOR with the hook positions
            key ^= zobristTable[ hook_pos[0][0] ][ hook_pos[0][1] ][ 'h1' ]
            key ^= zobristTable[ hook_pos[1][0] ][ hook_pos[1][1] ][ 'h2' ]
            
            # check table for state
            global lookUpTable
            if key in lookUpTable.keys():
                #print("used look up!!!!!!!!!!!!")
                return lookUpTable[key]
            else:
                # compute score and store with key in look up table
            
                # evaluation function
                score_diff = 0
                
                for f in fish_pos.keys():
                    if f in caught and (scores[0] + fish_score[f]) > max_score/2:
                        score_diff += 10000000
                    elif fish_score[f] < 0 and (hook_pos[0] == fish_pos[f]): 
                        score_diff += fish_score[f]
                    else:   
                        score_diff += (fish_score[f] / (1 + min((hook_pos[0][0] - fish_pos[f][0]) ** 2, (hook_pos[0][0] - fish_pos[f][0] - 20) ** 2) + (hook_pos[0][1] - fish_pos[f][1]) ** 2)
                                     - fish_score[f] / (1 + min((hook_pos[1][0] - fish_pos[f][0]) ** 2, (hook_pos[1][0] - fish_pos[f][0] - 20) ** 2) + (hook_pos[1][1] - fish_pos[f][1]) ** 2))
                
                #score_diff = sum([(fish_score[f] / (1 + shortest_distance_squared(hook_pos[0], fish_pos[f])) - fish_score[f] / (1 + shortest_distance_squared(hook_pos[1], fish_pos[f]))) for f in fish_pos.keys()])
                
                # print("computed value---------------")
                score = score_diff + scores[0] - scores[1]
                # store in look up table
                lookUpTable[key] = score
                return score
            """

            # without RSC
            # evaluation function
            score_diff = 0
            
            for f in fish_pos.keys():
                if f in caught and (scores[0] + fish_score[f]) > max_score/2:
                    score_diff = 10000000
                elif fish_score[f] < 0 and (hook_pos[0] == fish_pos[f]): 
                    score_diff += fish_score[f]
                else:   
                    score_diff += (fish_score[f] / (1 + min((hook_pos[0][0] - fish_pos[f][0]) ** 2, (hook_pos[0][0] - fish_pos[f][0] - 20) ** 2) + (hook_pos[0][1] - fish_pos[f][1]) ** 2)
                                    - fish_score[f] / (1 + min((hook_pos[1][0] - fish_pos[f][0]) ** 2, (hook_pos[1][0] - fish_pos[f][0] - 20) ** 2) + (hook_pos[1][1] - fish_pos[f][1]) ** 2))
            
            score = score_diff + scores[0] - scores[1]
            return score
            """
            
                

        elif state.get_player() == 0:
            v = -float('inf')
            for child in children:
                v = max(v, self.alphabeta(child, depth - 1, alpha, beta))
                alpha = max(alpha, v)
                if beta <= alpha:
                    break
        else:
            v = float('inf')
            for child in children:
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
        child_v = []
        prevBestMovesVal = -100000
        bestMove = 0
        # iterative deepening
        global start_time
        start_time = time.time()
        global time_out
        time_out = False
        depth_level = 2

        #print(len(lookUpTable))
        
        while not time_out:
            child_v = []
            for child in child_nodes:
                child_v.append(self.alphabeta(child, depth_level, -float('inf'), float('inf')))
            
            # check if partially checked tree returns better result then previous iteration
            max_val_of_current_iteration = max(child_v)
            
            if max_val_of_current_iteration > prevBestMovesVal:
                prevBestMovesVal = max_val_of_current_iteration
                bestMove = (child_nodes[child_v.index(max_val_of_current_iteration)]).move
            
            depth_level += 1
        
        #print(depth_level)
        return ACTION_TO_STR[bestMove]
