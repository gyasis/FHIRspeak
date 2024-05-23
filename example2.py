import os
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor import TreeRetriever, TreeRetrieverConfig
from raptor import BaseEmbeddingModel, ClusterTreeConfig
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
    # Configuration for the TreeBuilder
    tb_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Specify your tokenizer
    tb_max_tokens = 2000  # Set max tokens to 2000
    tb_num_layers = 5  # Aim to build 5 layers
    tb_threshold = 0.5
    tb_top_k = 5
    tb_selection_mode = "top_k"
    tb_summarization_length = 100
    tb_embedding_models = {"SBert": SBertEmbeddingModel()}
    tb_cluster_embedding_model = "SBert"

    # Create a TreeBuilderConfig
    tree_builder_config = ClusterTreeConfig(
        tokenizer=tb_tokenizer,
        max_tokens=tb_max_tokens,
        num_layers=tb_num_layers,
        threshold=tb_threshold,
        top_k=tb_top_k,
        selection_mode=tb_selection_mode,
        summarization_length=tb_summarization_length,
        embedding_models=tb_embedding_models,
        cluster_embedding_model=tb_cluster_embedding_model,
    )

    # Configuration for the RetrievalAugmentation
    ra_config = RetrievalAugmentationConfig(
        tree_builder_config=tree_builder_config
    )

    # Initialize the RetrievalAugmentation with the RetrievalAugmentationConfig
    ra = RetrievalAugmentation(config=ra_config)
    ra.tree = None  # Ensure the tree is initialized as None at the beginning

    # Process each file in the folder
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".txt"):  # Assuming all text files end with .txt
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r") as file:
                file_content = file.read()
                # Add the content of the file to the tree
                if ra.tree is None:
                    ra.add_documents(file_content)
                else:
                    ra.add_to_existing(file_content)
    
    # Detect the number of layers in the constructed tree
    actual_num_layers = ra.tree.num_layers + 1
    print(f"Actual number of layers in the tree: {actual_num_layers}")

    # Configuration for the TreeRetriever
    tr_tokenizer = tb_tokenizer  # Use the same tokenizer
    tr_threshold = tb_threshold
    tr_top_k = tb_top_k
    tr_selection_mode = tb_selection_mode
    tr_context_embedding_model = tb_cluster_embedding_model
    tr_embedding_model = tb_embedding_models["SBert"]
    tr_num_layers = actual_num_layers  # Match the number of layers in your tree
    tr_start_layer = 0  # Start from the first layer
    
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
    
    # Update the RetrievalAugmentation with the new TreeRetrieverConfig
    ra.tree_retriever_config = tree_retriever_config
    ra.retriever = TreeRetriever(tree_retriever_config, ra.tree)
    
    # Retrieve information from the tree based on the query
    information = ra.retrieve(question=query, top_k=10, max_tokens=3500, collapse_tree=True, return_layer_information=True)
    
    print(information)

if __name__ == "__main__":
    folder_path = "/home/gyasis/Documents/code/FHIRspeak/Flatten_patients/Flatten_Bao544_MacGyver246_a0b63e97-b6fd-5fe1-8f2d-2bec915efa97"  # Path to your folder containing text files
    query = "What is the main topic of these documents?"  # Your query
    process_folder(folder_path, query)
