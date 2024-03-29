import os
import random
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import nltk

# Only need to run once (uncomment if running locally)
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

# template user input 
user_input = "I love watching eagle,cat and bee and lalallalalalalalalal"

# Find animal nouns 
def find_animal_nouns(text):
    animal_list = ['cat', 'eagle', 'lion', 'bat']  # update in the future with the animal list
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    noun_words = [word for word, tag in tagged if tag in ('NN', 'NNS') and word in animal_list]
    return noun_words

# find out the path of the photos
def search_photos(animal_names):
    photo_paths = {}
    base_path = os.path.join('Auto-Segment-Collage', 'data', 'animals') # adjust it according to the system (linux/windows)
    for animal in animal_names:
        animal_path = os.path.join(base_path, animal)
        #print(animal_path)
        # if the animal exist
        if os.path.isdir(animal_path):
            photos = os.listdir(animal_path)
            # Filter for jpg photos
            jpg_photos = [photo for photo in photos if photo.endswith('.jpg')]
            if jpg_photos:
                # Randomly select one photo path
                selected_photo = random.choice(jpg_photos)
                photo_paths[animal] = os.path.join(animal_path, selected_photo)
    return photo_paths

print("Current Working Directory:", os.getcwd())

animal_nouns = find_animal_nouns(user_input)
# Test result queries
# print(animal_nouns)
selected_photos = search_photos(animal_nouns)
print(selected_photos)
