# %%


import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span
import networkx as nx
import matplotlib.pyplot as plt

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Create a matcher object
matcher = Matcher(nlp.vocab)

# Define the patterns to match
patterns = [
    [{"POS": "NOUN"}, {"POS": "VERB"}, {"POS": "NOUN"}],
    [{"POS": "NOUN"}, {"POS": "AUX"}, {"POS": "VERB"}, {"POS": "NOUN"}],
    [{"POS": "NOUN"}, {"POS": "VERB"}, {"POS": "ADP"}, {"POS": "NOUN"}],
    [{"POS": "NOUN"}, {"POS": "AUX"}, {"POS": "VERB"}, {"POS": "ADP"}, {"POS": "NOUN"}],
]

# Add the patterns to the matcher
matcher.add("SVO", patterns)

# Create a knowledge graph
G = nx.Graph()

# Specify the file path
file_path = "/home/gyasis/Documents/code/FHIRspeak/Flatten_patients/Flatten_Carrie738_Jeanene972_DuBuque211_6fbffc0b-7555-d8ea-d093-20517d5e4baf/Carrie738_Jeanene972_DuBuque211_6fbffc0b-7555-d8ea-d093-20517d5e4baf_28.txt"

# Open the file and read the content
with open(file_path, 'r') as file:
    text = file.read()

# Process the text
doc = nlp(text)

# Find the matches
matches = matcher(doc)

# Add the matches to the knowledge graph
for match_id, start, end in matches:
    span = doc[start:end]
    subject = span[0].text
    verb = span[1].text
    object = span[2].text

    # Add the nodes to the graph
    G.add_node(subject)
    G.add_node(object)

    # Add the edge to the graph
    G.add_edge(subject, object, label=verb)

# Draw the knowledge graph
nx.draw(G, with_labels=True)
plt.show()