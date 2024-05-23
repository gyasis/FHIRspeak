import os
import argparse
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

# Ensure the TEMP folder is created
def create_temp_folder():
    temp_folder = "TEMP"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    return temp_folder

def process_folder(folder_path):
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
    
    return ra

def save_tree(ra, save_path):
    ra.save(save_path)
    print(f"Tree successfully saved to {save_path}")

def load_tree(save_path):
    # Ensure SBertEmbeddingModel is used in TreeRetrieverConfig
    tb_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tb_threshold = 0.5
    tb_top_k = 15
    tb_selection_mode = "top_k"
    tb_embedding_models = {"SBert": SBertEmbeddingModel()}
    tb_cluster_embedding_model = "SBert"

    tree_retriever_config = TreeRetrieverConfig(
        tokenizer=tb_tokenizer,
        threshold=tb_threshold,
        top_k=tb_top_k,
        selection_mode=tb_selection_mode,
        context_embedding_model=tb_cluster_embedding_model,
        embedding_model=tb_embedding_models["SBert"],
        num_layers=None,  # Match the number of layers in your tree
        start_layer=None,  # Start from the first layer
    )

    # Load the tree
    ra = RetrievalAugmentation(config=RetrievalAugmentationConfig(tree_retriever_config=tree_retriever_config), tree=save_path)
    print(f"Tree successfully loaded from {save_path}")
    return ra

def answer_questions(ra):
    while True:
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        answer = ra.answer_question(question=question)
        print("Answer:", answer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process or query a document tree.')
    parser.add_argument('--ask', action='store_true', help='Load tree and ask questions.')
    args = parser.parse_args()

    temp_folder = create_temp_folder()
    save_path = os.path.join(temp_folder, "cinderella_tree")

    if args.ask:
        # Load the tree and answer questions
        ra = load_tree(save_path)
        answer_questions(ra)
    else:
        folder_path = "/home/gyasis/Documents/code/FHIRspeak/Flatten_patients/Flatten_Rasheeda241_Mirian768_Stanton715_4113255f-4e35-506a-ddef-4429caa17ffc"  # Path to your folder containing text files
        # Build and save the tree
        ra = process_folder(folder_path)
        save_tree(ra, save_path)

