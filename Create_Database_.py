import glob
import json
import os
import re

from pprint import pprint

from langchain.llms import Ollama
from langchain.graphs import Neo4jGraph
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOllama
from langchain import PromptTemplate

# Helper functions for resource to node and edges conversion
from FHIR_to_graph import resource_to_node, resource_to_edges

# Set environment variables
NEO4J_URI = os.getenv('FHIR_GRAPH_URL')
USERNAME = os.getenv('FHIR_GRAPH_USER')
PASSWORD = os.getenv('FHIR_GRAPH_PASSWORD')
DATABASE = os.getenv('FHIR_GRAPH_DATABASE')

# Establish Neo4j database connection
graph = Neo4jGraph(NEO4J_URI, USERNAME, PASSWORD, DATABASE)

# Define paths
working_dir = './working/bundles/'
synthea_bundles = glob.glob(f"{working_dir}/*.json")
synthea_bundles.sort()

# Initialize containers
nodes = []
edges = []
dates = set()  # Use set to ensure unique dates

# Load and process FHIR bundles
for bundle_file_name in synthea_bundles:
    with open(bundle_file_name) as raw:
        bundle = json.load(raw)
        for entry in bundle['entry']:
            resource = entry['resource']
            resource_type = resource['resourceType']
            if resource_type != 'Provenance':
                nodes.append(resource_to_node(resource))
                node_edges, node_dates = resource_to_edges(resource)
                edges.extend(node_edges)
                dates.update(node_dates)

# Create nodes for resources
for node in nodes:
    graph.query(node)

# Date pattern for node creation
date_pattern = re.compile(r'([0-9]+)/([0-9]+)/([0-9]+)')

# Create nodes for dates
for date in dates:
    date_parts = date_pattern.findall(date)[0]
    cypher_date = f'{date_parts[2]}-{date_parts[0]}-{date_parts[1]}'
    cypher = f'CREATE (:Date {{name:"{date}", id: "{date}", date: date("{cypher_date}")}})'
    graph.query(cypher)

# Create edges
for edge in edges:
    try:
        graph.query(edge)
    except Exception as e:
        print(f'Failed to create edge: {edge}\nError: {e}')

# Create the Vector Embedding Index in the Graph
Neo4jVector.from_existing_graph(
    HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5"),
    url=NEO4J_URI,
    username=USERNAME,
    password=PASSWORD,
    database=DATABASE,
    index_name='fhir_text',
    node_label="resource",
    text_node_properties=['text'],
    embedding_node_property='embedding',
)

# Create Vector Index
vector_index = Neo4jVector.from_existing_index(
    HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5"),
    url=NEO4J_URI,
    username=USERNAME,
    password=PASSWORD,
    database=DATABASE,
    index_name='fhir_text'
)

# Setup prompt templates
default_prompt = '''
System: Use the following pieces of context to answer the user's question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
{context}
Human: {question}
'''

prompt = PromptTemplate.from_template(default_prompt)

# Pick the LLM model to use
ollama_model = 'mistral'  # mistral, orca-mini, llama2

# Set K Nearest
k_nearest = 200

# Function to get date from question
def date_for_question(question_to_find_date, model):
    _llm = Ollama(model=model)
    _response = _llm(f'''
    system:Given the following question from the user, extract the date the question is asking about.
    Return the answer formatted as JSON only, as a single line.
    Use the form:
    
    {{"date":"[THE DATE IN THE QUESTION]"}}
    
    Use the date format of month/day/year.
    Use two digits for the month and day.
    Use four digits for the year.
    So 3/4/23 should be returned as {{"date":"03/04/2023"}}.
    So 04/14/89 should be returned as {{"date":"04/14/1989"}}.
    
    Please do not include any special formatting characters, like new lines or "\\n".
    Please do not include the word "json".
    Please do not include triple quotes.
    
    If there is no date, do not make one up. 
    If there is no date return the word "none", like: {{"date":"none"}}
    
    user:{question_to_find_date}
    ''')
    date_json = json.loads(_response)
    return date_json['date']

# Create contextualized vector store with date
def create_contextualized_vectorstore_with_date(date_to_look_for):
    if date_to_look_for == 'none':
        contextualize_query_with_date = """
        match (node)<-[]->(sc:resource)
        with node.text as self, reduce(s="", item in collect(distinct sc.text) | s + "\n\nSecondary Entry:\n" + item ) as ctxt, score, {} as metadata limit 1
        return "Primary Entry:\n" + self + ctxt as text, score, metadata
        """
    else:
        contextualize_query_with_date = f"""
        match (node)<-[]->(sc:resource)
        where exists {{
             (node)-[]->(d:Date {{id: '{date_to_look_for}'}})
        }}
        with node.text as self, reduce(s="", item in collect(distinct sc.text) | s + "\n\nSecondary Entry:\n" + item ) as ctxt, score, {{}} as metadata limit 1
        return "Primary Entry:\n" + self + ctxt as text, score, metadata
        """
    
    _contextualized_vectorstore_with_date = Neo4jVector.from_existing_index(
        HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5"),
        url=NEO4J_URI,
        username=USERNAME,
        password=PASSWORD,
        database=DATABASE,
        index_name='fhir_text',
        retrieval_query=contextualize_query_with_date,
    )
    return _contextualized_vectorstore_with_date

# Main function to ask question
def ask_date_question(question_to_ask, model=ollama_model, prompt_to_use=prompt):
    _date_str = date_for_question(question_to_ask, model)
    _index = create_contextualized_vectorstore_with_date(_date_str)
    _vector_qa = RetrievalQA.from_chain_type(
        llm=ChatOllama(model=model), chain_type="stuff",
        retriever=_index.as_retriever(search_kwargs={'k': k_nearest}),
        verbose=True,
        chain_type_kwargs={"verbose": True, "prompt": prompt_to_use}
    )
    return _vector_qa.run(question_to_ask)

# Example questions
questions = [
    "How much did the colon scan on Jan. 18, 2014 cost?",
    "What was the name of the patient whose respiratory rate was captured on 2/26/2017?",
    "Based on this explanation of benefits created on January 18, 2014, how much did it cost and what service was provided?"
]

for question in questions:
    print(ask_date_question(question))
