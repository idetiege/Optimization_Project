import matplotlib.pyplot as plt
from collections import Counter

def count_characters(text):
    # Define the valid characters
    valid_chars = "abcdefghijklmnopqrstuvwxyz.,?'"
    
    # Normalize text to lowercase and filter only valid characters
    filtered_text = [char.lower() for char in text if char.lower() in valid_chars]
    
    # Count character occurrences
    char_count = Counter(filtered_text)
    
    # Ensure all valid characters are present in the count (even if zero)
    for char in valid_chars:
        char_count.setdefault(char, 0)

    return char_count

def plot_character_count(char_count):
    # Sort data by frequency (most to least)
    sorted_data = sorted(char_count.items(), key=lambda x: x[1], reverse=True)
    characters, counts = zip(*sorted_data)

    # Plot using Matplotlib
    plt.figure(figsize=(12, 6))
    plt.bar(characters, counts, color='skyblue')
    plt.xlabel('Characters')
    plt.ylabel('Count')
    plt.title('Character Frequency (Most to Least)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def main():
    # Enter your paragraph here
    text = """The sun's warm glow fell across the field. A breeze stirred, rustling leaves as birds chirped. \
        The dog's bark echoed while a cat lounged nearby. People walked along quiet paths, sharing thoughts. \
        What joy exists in moments like these? Clouds drifted above, shadows shifting below. Foxes dashed through the brush. \
        Time's passage often feels swift. Yet, laughter lingers. Jars of jam lined the shelf. Vivid quilts hung, displaying vibrant hues. \
        Zebras grazed in far-off lands. Quirky scenes unfold daily. Few question why. \
        Life's charm, both simple and profound, remains constant. Is there anything more precious than this?"""
    char_count = count_characters(text)

    # Print results
    for char, count in sorted(char_count.items()):
        print(f"'{char}': {count}")

    # Plot the results
    plot_character_count(char_count)

if __name__ == "__main__":
    main()
