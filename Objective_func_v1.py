import numpy as np
import matplotlib.pyplot as plt

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
    
def objective_function(x): #take in x, a 30x2 matrix (30 rows for different keys, 2 columns for x and y positions)
    #initialize resting finger positions - assume that each finger comes to rest after pressing a key UNLESS the next key is pressed by the same finger
    f0 = np.array([1.5,2]) #pinky finger
    f1 = np.array([3.5,2.5]) #ring finger
    f2 = np.array([5.5,2.5]) #middle finger
    f3 = np.array([7.5,2.5]) #index finger
    fingers = np.array([f0,f1,f2,f3])
    
    #take in some string of letters (paragraph from chatgpt)
    string = "The sun's warm glow fell across the field. A breeze stirred, rustling leaves as birds chirped. \
        The dog's bark echoed while a cat lounged nearby. People walked along quiet paths, sharing thoughts. \
        What joy exists in moments like these? Clouds drifted above, shadows shifting below. Foxes dashed through the brush. \
        Time's passage often feels swift. Yet, laughter lingers. Jars of jam lined the shelf. Vivid quilts hung, displaying vibrant hues. \
        Zebras grazed in far-off lands. Quirky scenes unfold daily. Few question why. \
        Life's charm, both simple and profound, remains constant. Is there anything more precious than this?"
    string_index = string_to_index(string) #convert string to index in function above
    # print(string_index)
    total_dist = 0 # initialize total_dist to 0
    active_finger_prev = 4 # initialize active_finger_prev to a value that is not 0,1,2,3
    for j in range(len(string_index)):
        active_key = string_index[j]
        #find which finger is closest to the key
        if x[active_key][0] <= 2:
            active_finger = 0
        elif x[active_key][0] <= 4:
            active_finger = 1
        elif x[active_key][0] <= 6:
            active_finger = 2
        else:
            active_finger = 3
            
        #calculate the position of the active finger
        if j != 0 and active_finger == active_finger_prev:  
            finger_pos = finger_pos_prev
        else:
            finger_pos = fingers[active_finger]
        
        #reset active_finger_prev and finger_pos_prev
        active_finger_prev = active_finger
        finger_pos_prev = x[active_key]
        
        #calculate the distance between the active finger and the key
        key_dist = np.linalg.norm(finger_pos - x[active_key])
        total_dist += key_dist
    return total_dist

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




# def genetic_algorithm(f, num_people):
#     # Generate population
#     pop_1 = generate_population(num_people[0], {})  # Generate the first population without any fixed keys

#     # Evaluate fitness of each individual
#     fitness_scores_1 = [f(p) for p in pop_1]

#     # Sort individuals by fitness
#     sort_pop_1 = [x for _, x in sorted(zip(fitness_scores_1, pop_1))]

#     # Select the best individual (lowest distance)
#     best_indiv_1 = sort_pop_1[0]

#     # Find the index of best_individual in the population
#     best_index_1 = next(i for i, individual in enumerate(pop_1) if np.array_equal(individual, best_indiv_1))
#     best_score_1 = fitness_scores_1[best_index_1]


#     # Generate a new population with the fixed keys
#     fixed_keys = {k: best_indiv_1[k] for k in [0, 4, 8, 14, 19]}
#     pop_2 = generate_population(num_people[1], fixed_keys)  # Generate the second population with fixed keys

#     # Evaluate fitness of the new population
#     fitness_scores_2 = [f(p) for p in pop_2]

#     # Sort individuals by fitness
#     sort_pop_2 = [x for _, x in sorted(zip(fitness_scores_2, pop_2))]

#     # Select the best individual (lowest distance) from the new population
#     best_indiv_2 = sort_pop_2[0]

#     # Find the index of best_individual_2 in the new population
#     best_index_2 = next(i for i, individual in enumerate(pop_2) if np.array_equal(individual, best_indiv_2))
#     best_score_2 = fitness_scores_2[best_index_2]



#     return best_indiv_1, best_score_1, best_indiv_2, best_score_2  # Return the best individuals and their scores



def genetic_algorithm(f, num_people):
    best_individuals = []
    best_scores = []

    # Start with no fixed keys for the first generation
    fixed_keys = {}
    
    # Group indices into sets of 5
    fixed_key_indices = [
        [0, 4, 8, 14, 19],      # a, e, i, o, t
        [27, 26, 13, 18, 29],   # ,, ., n, s, '
        [7, 11, 3, 2, 17],      # h, l, d, c, r
        [12, 5, 15, 6, 20],     # m, f, p, g, u
        [24, 1, 21, 10, 22],    # y, b, v, k, w
        [9, 16, 25, 28, 23]     # j, q, z, ?, x
    ]

    for i in range(6):
        # Generate population with current fixed keys
        population = generate_population(num_people[i], fixed_keys)

        # Evaluate fitness
        fitness_scores = [f(p) for p in population]

        # Select the best individual
        best_index = np.argmin(fitness_scores)
        best_individual = population[best_index]
        best_score = fitness_scores[best_index]

        # Save results
        best_individuals.append(best_individual)
        best_scores.append(best_score)

        # Update fixed keys using the next set of 5 keys
        current_fixed_indices = fixed_key_indices[i]  # Get the next set of indices
        for idx in current_fixed_indices:
            fixed_keys[idx] = best_individual[idx]  # Fix the key to its position

    return best_individuals, best_scores









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





# print(population)

# # Extract x and y coordinates
# x_coords = person[:, 0]
# y_coords = person[:, 1]

# # Create scatter plot
# plt.scatter(x_coords, y_coords)

# # Add labels and title
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.title('Scatter Plot of Generated Individual')

# # Show plot
# plt.show()



# number_of_people = np.array([1000, 1000, 1000, 1000, 100, 100])
# best_individual_1, best_score_1, best_individual_2, best_score_2 = genetic_algorithm(objective_function, number_of_people)
# # print("Best Individual 1:", best_individual_1)
# print("Best Score 1:", best_score_1)  # This is the score of the first best individual
# # print("Best Individual 2:", best_individual_2)
# print("Best Score 2:", best_score_2)  # This is the score of the second best individual


# print_keyboard_layout(best_individual_1)  # Print the keyboard layout of the first best individual
# print_keyboard_layout(best_individual_2)




number_of_people = np.array([1000000, 100000, 100000, 100000, 100000, 120])
best_individuals, best_scores = genetic_algorithm(objective_function, number_of_people)
# Print all 6 scores
for i, score in enumerate(best_scores):
    print(f"Best Score {i + 1}: {score}")

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