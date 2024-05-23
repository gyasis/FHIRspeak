# Required imports
import spacy
import medspacy
from medspacy.visualization import visualize_ent
from openie import StanfordOpenIE
import networkx as nx
import matplotlib.pyplot as plt

# Specify the model name
model_name = "en_core_med7_trf"

# Load the medspacy model
nlp_medspacy = load_medspacy_model(model_name)
# Load the text file
file_path = "/home/gyasis/Documents/code/FHIRspeak/Flatten_patients/Flatten_Carrie738_Jeanene972_DuBuque211_6fbffc0b-7555-d8ea-d093-20517d5e4baf/Carrie738_Jeanene972_DuBuque211_6fbffc0b-7555-d8ea-d093-20517d5e4baf_28.txt"

with open(file_path, 'r') as file:
    text = file.read()

# Perform NER and visualize the results
doc_medspacy = nlp_medspacy(text)
print("Entities:")
for ent in doc_medspacy.ents:
    print(ent)

# Visualize the entities in the document
visualize_ent(doc_medspacy)

# Wait for a key press to continue
input("Press any key to continue to the knowledge graph construction...")

# Load spaCy's English model for further processing
nlp_spacy = spacy.load("en_core_web_sm")

# Perform NER with spaCy
doc_spacy = nlp_spacy(text)
entities = [(ent.text, ent.label_) for ent in doc_spacy.ents]
print("Named Entities with spaCy:", entities)

# OpenIE for extracting triples
with StanfordOpenIE() as client:
    triples = client.annotate(text)
    cleaned_triples = [{'subject': triple['subject'].lower(), 'relation': triple['relation'].lower(), 'object': triple['object'].lower()} for triple in triples]
    for triple in cleaned_triples:
        print(f"The {triple['subject']} {triple['relation']} {triple['object']}")
        print(f"Type of subject: {type(triple['subject'])}")
        print(f"Type of relation: {type(triple['relation'])}")
        print(f"Type of object: {type(triple['object'])}")

# Create a directed graph
knowledge_graph = nx.DiGraph()

# Add nodes and edges from cleaned triples
for triple in cleaned_triples:
    knowledge_graph.add_edge(triple['subject'], triple['object'], relation=triple['relation'])

# Add NER edges to the graph
for ent in entities:
    entity_text, entity_label = ent
    entity_text = entity_text.lower()
    knowledge_graph.add_node(entity_text, label=entity_label)

# Draw the graph
pos = nx.spring_layout(knowledge_graph)
nx.draw(knowledge_graph, pos, with_labels=True, arrows=True)

# Draw edge labels
edge_labels = nx.get_edge_attributes(knowledge_graph, 'relation')
nx.draw_networkx_edge_labels(knowledge_graph, pos, edge_labels=edge_labels)

plt.show()
