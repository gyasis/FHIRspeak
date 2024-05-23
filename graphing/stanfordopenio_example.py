# %%

from openie import StanfordOpenIE
import spacy
import networkx as nx
import matplotlib.pyplot as plt

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Specify the file path
file_path = "/home/gyasis/Documents/code/FHIRspeak/Flatten_patients/Flatten_Carrie738_Jeanene972_DuBuque211_6fbffc0b-7555-d8ea-d093-20517d5e4baf/Carrie738_Jeanene972_DuBuque211_6fbffc0b-7555-d8ea-d093-20517d5e4baf_28.txt"

# Open the file and read the content
with open(file_path, 'r') as file:
    text = file.read()

with StanfordOpenIE() as client:
    triples = client.annotate(text)
    for triple in triples:
        print(triple)
    
    cleaned_triples = [{'subject': triple['subject'].lower(), 'relation': triple['relation'].lower(), 'object': triple['object'].lower()} for triple in triples]
    
    for triple in cleaned_triples:
        print(f"The {triple['subject']} {triple['relation']} {triple['object']}")
        print(f"Type of subject: {type(triple['subject'])}")
        print(f"Type of relation: {type(triple['relation'])}")
        print(f"Type of object: {type(triple['object'])}")

# Perform NER on the text
doc = nlp(text)
entities = [(ent.text, ent.label_) for ent in doc.ents]
print("Named Entities:", entities)

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
# %%
