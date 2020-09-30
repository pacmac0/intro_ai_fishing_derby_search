#!/usr/bin/env python3
from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR
import random
import time

global start_time
global heuristic_tree


def shortest_distance_squared(pos1, pos2):
    return min((pos1[0] - pos2[0]) ** 2, (pos1[0] - pos2[0] - 20) ** 2) + (pos1[1] - pos2[1]) ** 2


def computeHash(fish_pos, hook_pos, ZobristTable):
    h = 0
    for fish in fish_pos:
        id = fish + 1
        h ^= ZobristTable[fish_pos[fish][0]][fish_pos[fish][1]][id]
    for hook in hook_pos:
        h ^= ZobristTable[hook_pos[hook][0]][hook_pos[hook][1]][hook]
    return h


def addHookHash(hook_pos, prev_hash, ZobristTable):
    h = prev_hash
    for hook in hook_pos:
        h ^= ZobristTable[hook_pos[hook][0]][hook_pos[hook][1]][hook]
    return h


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
        # Initializes hash table
        ZobristTable = [[[None for _ in range(len(initial_data) + 1)] for _ in range(20)] for _ in range(20)]
        for i in range(len(ZobristTable)):
            for j in range(len(ZobristTable[0])):
                for k in range(len(ZobristTable[0][0])):
                    ZobristTable[i][j][k] = random.randint(0, 3000)
        # EDIT THIS METHOD TO RETURN A MINIMAX MODEL ###
        return ZobristTable

    def alphabeta(self, ZobristTable, node, depth, alpha, beta, prevChildHash=None):

        global start_time
        state = node.state
        children = node.compute_and_get_children()
        if depth == 0 or (not children) or (time.time() - start_time > 0.064):

            global heuristic_tree
            if prevChildHash:
                hashValue = addHookHash(node.state.get_hook_positions(), prevChildHash, ZobristTable)
            else:
                hashValue = computeHash(state.get_fish_positions(), state.get_hook_positions(), ZobristTable)

            if hashValue in heuristic_tree:
                return [heuristic_tree[hashValue], hashValue]

            fish_pos = state.get_fish_positions()
            fish_score = state.get_fish_scores()
            hook_pos = state.get_hook_positions()
            curr_scores = state.get_player_scores()

            fish_score_diff = sum([fish_score[f] / (1 + shortest_distance_squared(hook_pos[0], fish_pos[f]))
                                   - fish_score[f] / (1 + shortest_distance_squared(hook_pos[1], fish_pos[f]))
                                   for f in fish_pos.keys()])

            h_value = fish_score_diff + curr_scores[0] - curr_scores[1]
            heuristic_tree[hashValue] = h_value
            return [h_value, hashValue]

        else:

            # move ordering - the move same as previous move first
            children.sort(key=lambda x: (x.move - node.move) ** 2, reverse=False)

            v0 = None
            if depth == 1:
                first_child = children[0]
                try:
                    v0, prevChildHash = self.alphabeta(ZobristTable, first_child, 0, alpha, beta)
                    prevChildHash = addHookHash(first_child.state.get_hook_positions(), prevChildHash, ZobristTable)
                except:
                    pass

            if state.get_player() == 0:
                if v0:
                    v = v0
                else:
                    v = -1e6
                for child in children[1:]:
                    if depth == 1:
                        v = max(v, (self.alphabeta(ZobristTable, child, depth - 1, alpha, beta, prevChildHash))[0])
                    else:
                        v = max(v, self.alphabeta(ZobristTable, child, depth - 1, alpha, beta))
                    alpha = max(alpha, v)
                    if beta <= alpha:
                        break
            else:
                if v0:
                    v = v0
                else:
                    v = 1e6
                for child in children[1:]:
                    if depth == 1:
                        v = min(v, (self.alphabeta(ZobristTable, child, depth - 1, alpha, beta, prevChildHash))[0])
                    else:
                        v = min(v, self.alphabeta(ZobristTable, child, depth - 1, alpha, beta))
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
        global start_time
        global heuristic_tree
        heuristic_tree = {}
        for child in child_nodes:
            start_time = time.time()
            child_v.append(self.alphabeta(model, child, 2, -1e6, 1e6))
        bestMove = (child_nodes[child_v.index(max(child_v))]).move
        return ACTION_TO_STR[bestMove]
