import ollama
import chromadb
from langchain_community.document_loaders import PyPDFLoader

# documents = [
#   "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
#   "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
#   "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
#   "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
#   "Llamas are vegetarians and have very efficient digestive systems",
#   "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
# ]

file_path = "./example_data/restaurant_menu.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

client = chromadb.Client()
collection = client.create_collection(name="docs")

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# store each document in a vector embedding database
for i, d in enumerate(all_splits):
  # Extract the page content as a string from the Document object
  page_content = d.page_content
  response = ollama.embed(model="mxbai-embed-large", input=page_content)
  embeddings = response["embeddings"]
#   print(f"Embedding for doc {i}: {embeddings}")
  collection.add(
    ids=[str(i)],
    embeddings=embeddings,
    documents=[page_content]  # Store the text content
  )

input = "What's ingredients are used in Indian Style Caesar Salad?"

# generate an embedding for the input and retrieve the most relevant doc
response = ollama.embed(
  model="mxbai-embed-large",
  input=input
)
results = collection.query(
  query_embeddings=response["embeddings"],
  n_results=1
)
data = results['documents'][0][0]

print(data)
print("-----")

output = ollama.generate(
  model="gemma3:4b",
  prompt=f"Using this data: {data}. Respond to this prompt: {input}"
)

print(output['response'])