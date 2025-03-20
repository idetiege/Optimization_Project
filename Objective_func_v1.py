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

def generate_individual():
    #this code generates a random 30x2 matrix (30 rows for different keys, 2 columns for x and y positions)
    #x can be an int with values of 1, 2, 3, 4, 5, 6, 7, 8
    # y is an int with values of 1, 2, 3 if x is 1 or 2 and 1, 2, 3, 4 if x is 3, 4, 5, 6, 7, 8
    # x and y pairings cannot be repeated
    x = np.zeros((30,2))
    used_pairs = set()
    for i in range(30):
        while True:
            x_val = np.random.randint(1, 9)
            if x_val == 1 or x_val == 2:
                y_val = np.random.randint(1, 4)
            else:
                y_val = np.random.randint(1, 5)
            if (x_val, y_val) not in used_pairs:
                used_pairs.add((x_val, y_val))
                x[i][0] = x_val
                x[i][1] = y_val
                break
    return x

population = []
for i in range(10):
    person = generate_individual()
    population.append(person)
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

total_distance = objective_function(person)
print(total_distance)