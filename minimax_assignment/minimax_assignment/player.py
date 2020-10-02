#!/usr/bin/env python3
from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR
import random
import time

start_time = 0
time_limit = 20
time_limit1 = 57
heuristic_tree = {}
check_order = {0: [0, 1, 2, 3, 4],  # if last move is stay, check stay - up - down - left - right
               1: [1, 3, 4, 0, 2],  # if last move is up, check up - stay - left - right - down
               2: [2, 3, 4, 0, 1],  # if last move is down, check down - stay - left - right - up
               3: [3, 1, 2, 0, 4],  # if last move is left, check left - stay - up - down - right
               4: [4, 1, 2, 0, 3]}  # if last move is right, check right - stay - up - down - left


def shortest_distance_squared(pos1, pos2):
    horizontal_diff = abs(pos1[0] - pos2[0])
    if horizontal_diff <= 10:
        return horizontal_diff ** 2 + (pos1[1] - pos2[1]) ** 2
    else:
        return (20 - horizontal_diff) ** 2 + (pos1[1] - pos2[1]) ** 2


def computeHash(fish_pos, hook_pos, ZobristTable):
    h = 0
    origin = hook_pos[0]

    for fish, position in fish_pos.items():
        pos = [position[0] - origin[0], position[1] - origin[1]]
        pos = [p if p >= 0 else 20 + p for p in pos]
        h ^= ZobristTable[pos[0]][pos[1]][fish + 1]

    pos = [hook_pos[1][0] - origin[0], hook_pos[1][1] - origin[1]]
    pos = [p if p >= 0 else 20 + p for p in pos]
    h ^= ZobristTable[pos[0]][pos[1]][0]

    return h


def find_nearest_fish(fish_dist, fish_id, n):
    return [fish_id[fish_dist.index(d)] for d in (sorted(fish_dist, reverse=False))[:n]]


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
            if any(c is not None for c in node.state.get_caught()):
                global heuristic_tree
                if len(heuristic_tree) > 0:
                    heuristic_tree = {}
                    # print('heuristic tree updated')

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
        ZobristTable = [[[None] for _ in range(20)] for _ in range(20)]
        for i in range(len(ZobristTable)):
            for j in range(len(ZobristTable[0])):
                ZobristTable[i][j] = [random.randint(0, 100000) for _ in range(len(initial_data))]

        global heuristic_tree
        heuristic_tree = {}

        return ZobristTable

    def alphabeta(self, ZobristTable, node, depth, alpha, beta, beam=3):

        state = node.state
        children = node.compute_and_get_children()

        global s_time
        if (time.time() - s_time) >= time_limit * pow(10, -3) or depth == 0 or (not children):

            global heuristic_tree
            hashValue = computeHash(state.get_fish_positions(), state.get_hook_positions(), ZobristTable)
            if hashValue in heuristic_tree:
                return heuristic_tree[hashValue]

            fish_pos = state.get_fish_positions()
            fish_score = state.get_fish_scores()
            hook_pos = state.get_hook_positions()
            curr_scores = state.get_player_scores()

            fish_dist = [shortest_distance_squared(hook_pos[0], pos) for pos in fish_pos.values()]
            if len(fish_dist) > beam:
                nearest_fish_idx = find_nearest_fish(fish_dist, list(fish_pos.keys()), beam)
                fish_score_diff = sum([fish_score[f] / (1 + dist)
                                       - fish_score[f] / (1 + shortest_distance_squared(hook_pos[1], fish_pos[f]))
                                       for (f, dist) in zip(fish_pos.keys(), fish_dist) if f in nearest_fish_idx])
            else:
                fish_score_diff = sum([fish_score[f] / (1 + dist)
                                       - fish_score[f] / (1 + shortest_distance_squared(hook_pos[1], fish_pos[f]))
                                       for (f, dist) in zip(fish_pos.keys(), fish_dist)])

            h_value = fish_score_diff + curr_scores[0] - curr_scores[1]
            heuristic_tree[hashValue] = h_value
            return h_value

        else:

            # move ordering - the move same as previous move first
            children_move = [child.move for child in children]
            order = [idx for idx in check_order[node.move] if idx in children_move]
            children = [children[idx] for idx in [children_move.index(o) for o in order]]
            # children.sort(key=lambda x: (x.move - node.move) ** 2, reverse=False)

            if state.get_player() == 0:
                v = -1e6
                for child in children:
                    v = max(v, self.alphabeta(ZobristTable, child, depth - 1, alpha, beta))
                    alpha = max(alpha, v)
                    if beta <= alpha:
                        break
            else:
                v = 1e6
                for child in children:
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

        # iterative deepening
        # depth_level = 2
        prevBestMovesVal = -1e6
        # start_time = time.time()
        # while (time.time() - start_time) < time_limit * pow(10, -3):
        for depth_level in range(2, 10):
            child_v = []
            start_time = time.time()
            for child in child_nodes:
                global s_time
                s_time = time.time()
                child_v.append(self.alphabeta(model, child, depth_level, -1e6, 1e6))

            # check if partially checked tree returns better result then previous iteration
            max_val_of_current_iteration = max(child_v)

            if max_val_of_current_iteration > prevBestMovesVal:
                prevBestMovesVal = max_val_of_current_iteration
                bestChild = child_nodes[child_v.index(max_val_of_current_iteration)]

            if time.time() - start_time > time_limit1 * pow(10, -3):
                print(depth_level)
                break

            # child_v = [self.alphabeta(model, child, depth_level, -1e6, 1e6) for child in child_nodes]
            # bestChild = child_nodes[child_v.index(max(child_v))]
            # depth_level += 1
        # print(depth_level)
        return ACTION_TO_STR[bestChild.move]
