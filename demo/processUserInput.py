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
    animal_list = ['cat', 'eagle', 'lion', 'bat']  
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    noun_words = [word for word, tag in tagged if tag in ('NN', 'NNS') and word in animal_list]
    return noun_words


# Execution
animal_nouns = find_animal_nouns(user_input)
print(animal_nouns)
# selected_photos = search_photos(animal_nouns)
# print(selected_photos)
