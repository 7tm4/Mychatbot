import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# ✅ Set your Groq API key (optional — used only for text generation, not embeddings)
os.environ["GROQ_API_KEY"] = "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"

# Step 1: Load all .txt files from past_meetings/
docs = []
folder_path = "past_meetings"
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        loader = TextLoader(os.path.join(folder_path, filename))
        data = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs.extend(splitter.split_documents(data))

# Step 2: Embed and store in FAISS using HuggingFace Embeddings
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embedding)

# Step 3: Load your new transcript
with open("transcript.txt", "r", encoding="utf-8") as f:
    new_transcript = f.read()

# Step 4: Search for the top 2 most similar past meetings
results = db.similarity_search(new_transcript, k=2)

# Step 5: Print the results
print("\n🧠 Top 2 Related Past Meetings:\n")
for i, doc in enumerate(results, start=1):
    print(f"--- Result {i} ---\n{doc.page_content}\n")

