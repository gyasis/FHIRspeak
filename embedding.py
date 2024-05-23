from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
client = OpenAI()


# Load environment variables
load_dotenv(
    "/media/gyasis/Blade 15 SSD/Users/gyasi/Google Drive (not syncing)/Collection/chatrepository/.env",
    override=True,
)
embedding = OpenAIEmbeddings(disallohispered_special=())

def embedding_function(texts, model="text-embedding-ada-002"):
    if isinstance(texts, str):
        texts = [texts]
    texts = [t.replace("\n", " ") for t in texts]
    return [
        data["embedding"]
        for data in client.embeddings.create(input=texts, model=model)["data"]
    ]

