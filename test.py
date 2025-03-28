import numpy as np
import matplotlib.pyplot as plt
import heapq


import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, layout, fixed_keys, depth, cost, lower_bound):
        self.layout = layout
        self.fixed_keys = fixed_keys
        self.depth = depth
        self.cost = cost
        self.lower_bound = lower_bound

    def __lt__(self, other):
        return self.lower_bound < other.lower_bound

def string_to_index(string):
    string = string.lower().replace(" ", "")
    index = []
    for char in string:
        if char.isalpha():
            index.append(ord(char) - 97)
        elif char == '.':
            index.append(26)
        elif char == ',':
            index.append(27)
        elif char == '?':
            index.append(28)
        elif char == "'":
            index.append(29)
    return index

def objective_function(x):
    f0 = np.array([1.5, 2])
    f1 = np.array([3.5, 2.5])
    f2 = np.array([5.5, 2.5])
    f3 = np.array([7.5, 2.5])
    fingers = np.array([f0, f1, f2, f3])

    string = "The sun's warm glow fell across the field. A breeze stirred, rustling leaves as birds chirped."
    string_index = string_to_index(string)

    total_dist = 0
    active_finger_prev = 4
    for j in range(len(string_index)):
        active_key = string_index[j]
        if x[active_key][0] <= 2:
            active_finger = 0
        elif x[active_key][0] <= 4:
            active_finger = 1
        elif x[active_key][0] <= 6:
            active_finger = 2
        else:
            active_finger = 3

        finger_pos = fingers[active_finger] if j == 0 or active_finger != active_finger_prev else finger_pos_prev
        active_finger_prev = active_finger
        finger_pos_prev = x[active_key]

        key_dist = np.linalg.norm(finger_pos - x[active_key])
        total_dist += key_dist

    return total_dist

def estimate_lower_bound(layout, depth):
    # Simplified lower bound based on the remaining keys and average cost
    # print("Estimating Lower Bound...")
    remaining_keys = 30 - depth
    avg_distance = 2.0  # Assume a reasonable average distance
    return objective_function(layout) + remaining_keys * avg_distance

def branch_and_bound():
    print("Running Branch and Bound Algorithm...")
    initial_layout = np.zeros((30, 2))
    fixed_keys = {}
    initial_node = Node(initial_layout, fixed_keys, 0, 0, 0)
    best_cost = float('inf')
    best_layout = None
    pq = []
    heapq.heappush(pq, initial_node)

    while pq:
        node = heapq.heappop(pq)

        if node.depth == 30:  # All keys placed
            current_cost = objective_function(node.layout)
            if current_cost < best_cost:
                best_cost = current_cost
                best_layout = node.layout.copy()
        else:
            for x in range(1, 9):
                for y in range(1, 5 if x >= 3 else 4):
                    if (x, y) not in [tuple(pos) for pos in node.layout]:
                        new_layout = node.layout.copy()
                        new_layout[node.depth] = [x, y]
                        new_cost = objective_function(new_layout)
                        lower_bound = estimate_lower_bound(new_layout, node.depth + 1)

                        if lower_bound < best_cost:  # Bounding step
                            heapq.heappush(pq, Node(new_layout, node.fixed_keys, node.depth + 1, new_cost, lower_bound))

    return best_layout, best_cost

def print_keyboard_layout(x):
    layout = [[" " for _ in range(8)] for _ in range(4)]
    letters = "abcdefghijklmnopqrstuvwxyz.,?'"

    for i in range(30):
        x_pos = int(x[i][0]) - 1
        y_pos = 3 - (int(x[i][1]) - 1)
        layout[y_pos][x_pos] = letters[i]

    print("  1 2 3 4 5 6 7 8")
    print(" +----------------")
    for i in range(4):
        row = f"{4-i}| " + " ".join(layout[i])
        print(row)

best_layout, best_cost = branch_and_bound()
print(f"Best Cost: {best_cost}")
print("Optimal Keyboard Layout:")
print_keyboard_layout(best_layout)
