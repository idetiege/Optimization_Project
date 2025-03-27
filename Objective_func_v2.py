'''
The purpose of this code is...


Authors:
 - Austin Erickson "The Brain"
 - Isaac Detiege "The Muscle"
 - Ammon Miller "The Milkman" (Copilot generated, lol)

Last updated: 3/27/2025 around 3pm

Notes:
- Updated layout. Feel free to add / remove comments. I have no attachment to them. 
- If same_finger_penalty is really big, then e gets moved to the pinky finger. We theorize 
  that this is because lots of letters come before and after e.
- We are currently using the same number of people for each generation. We could change this
  to use a different number of people for each generation.
- We could also change the number of generations to run for. We are currently running for 6 generations.
- We could also change the number of fixed keys to use. We are currently using 5 fixed keys for each generation.
- We could also change the number of keys to use. We are currently using 30 keys.
- We could also change the number of fingers to use. We are currently using 4 fingers.
- We could also change the number of generations to run for. We are currently running for 6 generations.
- We could also change the number of people to use. We are currently using 10000 people for the first generation and 1000 people for the next 28 generations and 120 people for the last generation.
- We could also change the number of generations to run for. We are currently running for 6 generations.
- We could also change the number of people to use. We are currently using 10000 people for the first generation and 1000 people for the next 28 generations and 120 people for the last generation.

'''
#---------------------------------
# Import 
#---------------------------------
import numpy as np
import matplotlib.pyplot as plt


#---------------------------------
# Functions
#---------------------------------
def string_to_index(string):
    # Convert string to lower case
    string = string.lower()
    # Remove spaces from the string
    string = string.replace(" ", "")
    # Convert letters and punctuation in the string to index where a is 0, b is 1, etc.
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
    
def objective_function(x): #take in an individual, x, which is a 30x2 matrix (30 rows for different keys, 2 columns for x and y positions)
    #initialize resting finger positions - assume that each finger comes to rest after pressing a key UNLESS the next key is pressed by the same finger
    f0 = np.array([1.5,2]) #pinky finger
    f1 = np.array([3.5,2.5]) #ring finger
    f2 = np.array([5.5,2.5]) #middle finger
    f3 = np.array([7.5,2.5]) #index finger
    home_positions = np.array([f0,f1,f2,f3])

    #initialize counters
    pinky_finger_count = 0
    ring_finger_count = 0
    middle_finger_count = 0
    index_finger_count = 0
    same_finger_penalty_count = 0
    
    #take in some string of letters (paragraph from chatgpt)
    string = "The sun's warm glow fell across the field. A breeze stirred, rustling leaves as birds chirped. \
    The dog's bark echoed while a cat lounged nearby. People walked along quiet paths, sharing thoughts. \
    What joy exists in moments like these? Clouds drifted above, shadows shifting below. Foxes dashed through the brush. \
    Time's passage often feels swift. Yet, laughter lingers. Jars of jam lined the shelf. Vivid quilts hung, displaying vibrant hues. \
    Zebras grazed in far-off lands. Quirky scenes unfold daily. Few question why. \
    Life's charm, both simple and profound, remains constant. Is there anything more precious than this?"

    string_index = string_to_index(string) #convert string to index in function above    
    score = 0 # initialize total_dist to 0
    prev_active_finger = np.inf # initialize active_finger_prev to a value that is not 0,1,2,3
    for j in range(len(string_index)):

        # Where is the key?
        next_key = string_index[j]
        next_finger_position = x[next_key]

        if x[next_key][0] <= 2:   # pinky finger
            next_active_finger = 0
            active_finger_penalty = 100
            pinky_finger_count += 1
        elif x[next_key][0] <= 4: # ring finger
            next_active_finger = 1
            active_finger_penalty = 5
            ring_finger_count += 1
        elif x[next_key][0] <= 6: # middle finger
            next_active_finger = 2
            active_finger_penalty = 0
            middle_finger_count += 1
        else:                       # index finger
            next_active_finger = 3
            active_finger_penalty = 0
            index_finger_count += 1
            
        # Calc current_finger_position
        if j != 0 and next_active_finger == prev_active_finger:     # Same finger
            current_finger_position = prev_finger_position
            same_finger_penalty = 2000
            same_finger_penalty_count += 1
        else:                                                       # Different finger
            current_finger_position = home_positions[next_active_finger]
            same_finger_penalty = 0
        
        # Reset prev_finger_position and prev_active_finger
        prev_finger_position = next_finger_position
        prev_active_finger = next_active_finger
        
        # Calc distance
        distance = np.linalg.norm(current_finger_position - x[next_key])

        # Calc score
        score += distance + active_finger_penalty + same_finger_penalty

    # Combine counters
    counters = [pinky_finger_count, ring_finger_count, middle_finger_count, index_finger_count, same_finger_penalty_count]
    return score, counters

def generate_individual(fixed_keys):
    #this code generates a random 30x2 matrix (30 rows for different keys, 2 columns for x and y positions)
    #x can be an int with values of 1, 2, 3, 4, 5, 6, 7, 8
    # y is an int with values of 1, 2, 3 if x is 1 or 2 and 1, 2, 3, 4 if x is 3, 4, 5, 6, 7, 8
    # x and y pairings cannot be repeated
    
    # Initialize person with fixed keys
    person = np.zeros((30,2))
    used_pairs = set()

    for key, value in fixed_keys.items():
        person[key] = value
        used_pairs.add(tuple(value))

    for i in range(30):
        if i in fixed_keys:  # Skip fixed keys
            continue
        while True:
            x_val = np.random.randint(1, 9)         # Random x value between 1 and 8
            if x_val == 1 or x_val == 2:            # Pinky range
                y_val = np.random.randint(1, 4)     # Choose y value between 1 and 3
            else:                                   # Other fingers range
                y_val = np.random.randint(1, 5)     # Choose y value between 1 and 4

            if (x_val, y_val) not in used_pairs:
                used_pairs.add((x_val, y_val))
                person[i] = (x_val, y_val)
                break
    return person

def generate_population(num_people, fixed_keys):
    # Create a population of num_people persons
    population = []
    for i in range(num_people):
        person = generate_individual(fixed_keys)
        population.append(person)
    return population

def genetic_algorithm(f, num_people):
    best_individuals = []
    best_scores = []
    best_counters = []

    # Start with no fixed keys for the first generation
    fixed_keys = {}
    # Group indices into sets of 1
    fixed_key_indices = [
        [0],  # a
        [4],  # e
        [8],  # i
        [14], # o
        [19], # t
        [27], # ,
        [26], # .
        [13], # n
        [18], # s
        [29], # '
        [7],  # h
        [11], # l
        [3],  # d
        [2],  # c
        [17], # r
        [12], # m
        [5],  # f
        [15], # p
        [6],  # g
        [20], # u
        [24], # y
        [1],  # b
        [21], # v
        [10], # k
        [22], # w
        [9],  # j
        [16], # q
        [25], # z
        [28], # ?
        [23]  # x
    ]

    for i in range(30):
        # Generate population with current fixed keys
        population = generate_population(num_people[i], fixed_keys)

        # Evaluate fitness
        fitness_scores = [f(p) for p in population][0]

        # Select the best individual
        best_index = np.argmin(fitness_scores[0])
        best_individual = population[best_index]
        best_score = fitness_scores[best_index]
        best_counter = fitness_scores[1]

        # Save results
        best_individuals.append(best_individual)
        best_scores.append(best_score)
        best_counters.append(best_counter)

        # Update fixed keys using the next set of 5 keys
        current_fixed_indices = fixed_key_indices[i]  # Get the next set of indices
        for idx in current_fixed_indices:
            fixed_keys[idx] = best_individual[idx]  # Fix the key to its position

    return best_individuals, best_scores, best_counters


def print_keyboard_layout(x):
    layout = [[" " for _ in range(8)] for _ in range(4)]
    letters = "abcdefghijklmnopqrstuvwxyz.,?'"
    
    for i in range(30):
        x_pos = int(x[i][0]) - 1
        y_pos = 3 - (int(x[i][1]) - 1)  # Flip the y-axis
        layout[y_pos][x_pos] = letters[i]
    
    print("  1 2 3 4 5 6 7 8")
    print(" +----------------")
    for i in range(4):
        row = f"{4-i}| " + " ".join(layout[i])  # Adjust row numbering
        print(row)


#---------------------------------
# Optimize
#---------------------------------
number_of_people = np.array([10000] + [1000] * 28 + [120])
best_individuals, best_scores, best_counters = genetic_algorithm(objective_function, number_of_people)


#---------------------------------
# Print & Plot Results
#---------------------------------
# Print all 6 scores
for i, score in enumerate(best_scores):
    print(f"Best Score {i + 1}: {score}")

for i, counter in enumerate(best_counters):
    print(f"Counter {i + 1}: {counter}")

# Print all 6 keyboard layouts
for i, individual in enumerate(best_individuals):
    print(f"Keyboard Layout {i + 1}:")
    print_keyboard_layout(individual)
    print()

    # Create a convergence graph
    generations = range(1, len(best_scores) + 1)

plt.figure(figsize=(10, 6))
plt.plot(generations, best_scores, marker='o', linestyle='-', color='b')
plt.xlabel('Generation')
plt.ylabel('Best Score')
plt.title('Convergence Graph')
plt.grid(True)
plt.xticks(generations)  # Ensure all generations are labeled on the x-axis
plt.savefig('genetic_conv.png')  # Save the figure as genetic_conv.png
plt.show()