import os
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor import TreeRetriever, TreeRetrieverConfig
from raptor import BaseEmbeddingModel, ClusterTreeConfig
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer
import concurrent.futures

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

def process_file(filepath, config):
    with open(filepath, "r") as file:
        file_content = file.read()
    ra = RetrievalAugmentation(config=config)
    if os.path.exists("TEMP/temp_tree"):
        ra = RetrievalAugmentation(tree="TEMP/temp_tree")
    if ra.tree is None:
        ra.add_documents(file_content)
    else:
        ra.add_to_existing(file_content)
    ra.save("TEMP/temp_tree")

def process_folder_parallel(folder_path, config, batch_size=50):
    # Get a list of all text files in the folder
    filepaths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(".txt")]

    # Process files in smaller batches to avoid too many open files
    for i in tqdm(range(0, len(filepaths), batch_size), desc="Processing files"):
        batch = filepaths[i:i + batch_size]
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(batch), os.cpu_count())) as executor:
            list(tqdm(executor.map(process_file, batch, [config]*len(batch)), total=len(batch), desc="Batch progress"))

def save_tree(ra, save_path):
    ra.save(save_path)
    print(f"Tree successfully saved to {save_path}")

def load_tree(save_path):
    ra = RetrievalAugmentation(tree=save_path)
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
    folder_path = "/home/gyasis/Documents/code/FHIRspeak/Flatten_patients/Flatten_Bao544_MacGyver246_a0b63e97-b6fd-5fe1-8f2d-2bec915efa97"  # Path to your folder containing text files
    temp_folder = create_temp_folder()
    save_path = os.path.join(temp_folder, "cinderella_tree")
    
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

    # Process folder in parallel with batch size to avoid too many open files
    process_folder_parallel(folder_path, ra_config)
    
    # Load the tree and answer questions
    ra = load_tree("TEMP/temp_tree")
    answer_questions(ra)
