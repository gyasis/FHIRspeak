import os
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor import TreeRetriever, TreeRetrieverConfig
from raptor import BaseEmbeddingModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer

# Load environment variables
load_dotenv(
    "/media/gyasis/Blade 15 SSD/Users/gyasi/Google Drive (not syncing)/Collection/chatrepository/.env",
    override=True,
)

# Get the OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Define your own embedding model class
class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)

def process_folder(folder_path, query):
    # Configuration for the TreeRetriever
    tr_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Specify your tokenizer
    tr_threshold = 0.5
    tr_top_k = 5
    tr_selection_mode = "top_k"
    tr_context_embedding_model = "OpenAI"
    tr_embedding_model = SBertEmbeddingModel()  # Use your embedding model
    tr_num_layers = 2  # Adjust this value to match the number of layers in your tree
    tr_start_layer = 1  # Adjust this value to match the start layer of your tree
    
    # Create a TreeRetrieverConfig
    tree_retriever_config = TreeRetrieverConfig(
        tokenizer=tr_tokenizer,
        threshold=tr_threshold,
        top_k=tr_top_k,
        selection_mode=tr_selection_mode,
        context_embedding_model=tr_context_embedding_model,
        embedding_model=tr_embedding_model,
        num_layers=tr_num_layers,
        start_layer=tr_start_layer,
    )
    
    # Configuration for the RetrievalAugmentation
    ra_config = RetrievalAugmentationConfig(tree_retriever_config=tree_retriever_config)

    # Initialize the RetrievalAugmentation with the RetrievalAugmentationConfig
    ra = RetrievalAugmentation(config=ra_config)
    
    # Process each file in the folder
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".txt"):  # Assuming all text files end with .txt
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r") as file:
                file_content = file.read()
                # Add the content of the file to the tree
                ra.add_documents(file_content)
    
    # Retrieve information from the tree based on the query
    information = ra.retrieve(question=query, top_k=10, max_tokens=3500, collapse_tree=True, return_layer_information=True)
    
    print(information)

if __name__ == "__main__":
    folder_path = "/home/gyasis/Documents/code/FHIRspeak/Flatten_patients/Flatten_Bao544_MacGyver246_a0b63e97-b6fd-5fe1-8f2d-2bec915efa97"  # Path to your folder containing text files
    query = "What is the main topic of these documents?"  # Your query
    process_folder(folder_path, query)
