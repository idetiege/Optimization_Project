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

def layout_to_matrix(layout_str):
    # Split input string into rows
    rows = layout_str.strip().split("\n")
    
    # Create a dictionary to hold the positions of each key
    positions = {}
    
    # Row mappings to reverse the input format: rows 4, 3, 2, 1
    row_mapping = {4: 3, 3: 2, 2: 1, 1: 0}
    
    # Parse each row and column in the layout string
    for row in rows[1:]:
        # Skip the separator line containing '+----------------'
        if '+' in row:
            continue
        
        parts = row.split("|")  # Split by '|', the first part is the row label
        row_label = int(parts[0].strip())  # Row number (4, 3, 2, 1)
        keys = parts[1].strip().split()  # List of keys in the row
        
        # Reverse row number for coordinates and store each key position
        for col, key in enumerate(keys):
            if key:  # Only process non-empty keys
                positions[key] = (col, row_mapping[row_label])
    
    # Convert the dictionary of positions to a matrix 'p'
    max_x = max([pos[0] for pos in positions.values()])
    max_y = max([pos[1] for pos in positions.values()])
    
    # Create the output matrix with empty strings for unused spots
    p = np.full((max_y + 1, max_x + 1), "", dtype=object)
    
    # Populate the matrix with positions
    for key, (x, y) in positions.items():
        p[y, x] = key
    
    return p

def convert_layout_to_coordinates(layout_matrix):
    # Create a mapping of each key to its (x, y) coordinates
    key_coordinates = {}
    
    for y in range(layout_matrix.shape[0]):
        for x in range(layout_matrix.shape[1]):
            key = layout_matrix[y, x]
            if key:  # If there's a key at the position
                key_coordinates[key] = (x, y)
    
    # Now create a 30x2 matrix for the objective function
    result = np.zeros((30, 2))
    for i, key in enumerate("abcdefghijklmnopqrstuvwxyz.,?'"):
        if key in key_coordinates:
            result[i] = key_coordinates[key]
        else:
            result[i] = [-1, -1]  # In case a key doesn't exist in the layout
    
    return result

# Test the function with your input layout
layout_str = """
1 2 3 4 5 6 7 8
 +----------------
4|     a b c d e f
3| g h i j k l m n
2| o p q r s t u v
1| w x y z . , ? '
"""

# Get the layout matrix
layout_matrix = layout_to_matrix(layout_str)

# Convert the layout matrix into a 30x2 matrix for the objective function
coordinates = convert_layout_to_coordinates(layout_matrix)
print(coordinates)
p = np.array([[3,4],[4,4],[5,4],[6,4],[7,4],[8,4],[1,3],[2,3],[3,3],[4,3],[5,3],[6,3],
    [7,3],[8,3],[1,2],[2,2],[3,2],[4,2],[5,2],[6,2],[7,2],[8,2],[1,1],[2,1],
    [3,1],[4,1],[5,1],[6,1],[7,1],[8,1]])

# Now you can pass the coordinates into the objective function to get a score
score = objective_function(p)

# Print the score
print(f"Objective function score: {score}")
