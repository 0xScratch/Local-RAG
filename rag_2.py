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
    chunk_size=400, chunk_overlap=100, add_start_index=True
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

input = "What's the price of Bun Cha Gio and the ingredients involved in it? What can you tell me more about this recipe as well as the ingredients?"

# Process query to understand what information is being asked for
import re

def analyze_query(query):
    """
    Analyzes a query to determine what information is being asked for,
    and extracts entities like dish names.
    
    Returns:
        dict: Dictionary with keys:
            - dish_names: List of dish names found in query
            - is_price_query: Boolean indicating if query is about price
            - is_ingredient_query: Boolean indicating if query is about ingredients
            - is_nutrition_query: Boolean indicating if query is about nutrition
    """
    query_lower = query.lower()
    
    # Check what kind of information is being asked for
    is_price_query = any(term in query_lower for term in ['price', 'cost', 'how much', 'expensive'])
    is_ingredient_query = any(term in query_lower for term in ['ingredient', 'made of', 'contain', 'what is in'])
    is_nutrition_query = any(term in query_lower for term in ['calorie', 'nutrition', 'health', 'kcal', 'fat'])
    
    # Extract dish names
    # Remove common question patterns
    cleaned_query = re.sub(r"what'?s\s+the\s+(price|ingredients|nutrition)\s+of\s+", "", query_lower)
    cleaned_query = re.sub(r"how\s+much\s+does\s+(the|a)\s+", "", cleaned_query)
    cleaned_query = re.sub(r"(what are|tell me about)\s+(the)?\s*", "", cleaned_query)
    cleaned_query = re.sub(r"(ingredients|cost|price|nutrition).*", "", cleaned_query)
    
    # Extract potential dish names (capitalized phrases or phrases followed by "dish")
    potential_dishes = []
    
    # Add the main query without question words
    cleaned = re.sub(r'\b(what|how|price|cost|ingredients|involved|and|the|of|is|in)\b', '', cleaned_query)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if cleaned:
        potential_dishes.append(cleaned)
    
    # Find phrases in original query that might be dish names
    matches = re.findall(r'\b([A-Z][A-Za-z\s]+)\b', query)
    potential_dishes.extend(matches)
    
    # Find phrases followed by "dish"
    matches = re.findall(r'\b([A-Za-z\s]+)\s+dish\b', query_lower)
    potential_dishes.extend(matches)
    
    # Remove duplicates and clean
    dish_names = [dish.strip() for dish in potential_dishes if dish.strip()]
    
    return {
        "dish_names": dish_names,
        "is_price_query": is_price_query,
        "is_ingredient_query": is_ingredient_query,
        "is_nutrition_query": is_nutrition_query
    }

def extract_dish_names(query):
    """Legacy function for backward compatibility"""
    return analyze_query(query)["dish_names"]

# Process the query to understand what the user is asking about
query_analysis = analyze_query(input)
dish_names = query_analysis["dish_names"]
is_price_query = query_analysis["is_price_query"]
is_ingredient_query = query_analysis["is_ingredient_query"]
is_nutrition_query = query_analysis["is_nutrition_query"]

print(f"Looking for these dishes: {dish_names}")
print(f"Query type - Price: {is_price_query}, Ingredients: {is_ingredient_query}, Nutrition: {is_nutrition_query}")

# Generate embeddings for the input
response = ollama.embed(
  model="mxbai-embed-large",
  input=input
)

# Retrieve multiple relevant documents
results = collection.query(
  query_embeddings=response["embeddings"],
  n_results=5  # Get more results to ensure we find relevant info
)

# Function to score document relevance to dishes
def score_document_for_dishes(doc, dish_names):
    doc_lower = doc.lower()
    best_score = 0
    
    for dish in dish_names:
        dish_lower = dish.lower()
        words = dish_lower.split()
        
        # Score based on full dish name match
        if dish_lower in doc_lower:
            score = 100
        # Score based on individual word matches
        else:
            score = sum(10 for word in words if len(word) > 2 and word in doc_lower)
        
        # Boost if it has price indicators (stand-alone numbers)
        if re.search(r'\b\d{3,4}\b', doc):
            score += 30
            
        best_score = max(best_score, score)
    
    return best_score

# Function to extract price for a dish from text
def extract_price(doc, dish_name):
    """
    Attempts to extract the price for a specific dish from menu text
    Returns a tuple (price, confidence) where confidence is a score indicating
    how confident we are about the extracted price
    """
    dish_name_lower = dish_name.lower()
    doc_lower = doc.lower()
    
    # If dish name not in document, return nothing
    if dish_name_lower not in doc_lower:
        return None, 0
    
    # Get the position of the dish name
    dish_pos = doc_lower.find(dish_name_lower)
    
    # Find all standalone numbers that could be prices (3-4 digits)
    # But exclude numbers that appear in nutritional info patterns
    prices_with_positions = []
    
    # Find all potential prices (3-4 digit numbers)
    for match in re.finditer(r'\b(\d{3,4})\b', doc):
        price = match.group(1)
        position = match.start()
        
        # Check if this number is part of nutritional info
        surrounding_text = doc[max(0, position-10):min(len(doc), position+30)]
        
        # Skip if it looks like nutritional info with gms, kcal, etc.
        if re.search(r'\(\d+\s*gms|\d+\s*[Kk]cal|\d+\s*cal', surrounding_text):
            continue
            
        # Skip if it's followed by measurement units (indicating weight/volume)
        if re.search(r'\d+\s*(gms|ml|g|kg|oz)', surrounding_text):
            continue
        
        prices_with_positions.append((price, position))
    
    if not prices_with_positions:
        return None, 0
    
    # Look for the closest price BEFORE the dish name (higher priority)
    prices_before = [(price, pos) for price, pos in prices_with_positions if pos < dish_pos]
    if prices_before:
        # Find the closest price before the dish name
        closest_before = max(prices_before, key=lambda x: x[1])
        
        # If it's close to the dish name (within ~2 lines), it's likely the price
        if dish_pos - closest_before[1] < 100:
            return closest_before[0], 90
        # Otherwise, it's still a candidate but with lower confidence
        else:
            return closest_before[0], 70
    
    # Look for prices on the same line as the dish name or close by
    dish_line_start = doc[:dish_pos].rfind("\n")
    if dish_line_start == -1:
        dish_line_start = 0
        
    dish_line_end = doc[dish_pos:].find("\n")
    if dish_line_end == -1:
        dish_line_end = len(doc)
    else:
        dish_line_end += dish_pos
        
    # Expand to include adjacent lines
    dish_line_context = doc[max(0, dish_line_start-50):min(len(doc), dish_line_end+50)]
    
    # Find prices in this context
    prices_in_context = re.findall(r'\b(\d{3,4})\b', dish_line_context)
    if prices_in_context:
        for price in prices_in_context:
            # Skip prices that might be nutritional info
            if re.search(fr'{price}\s*gms|{price}\s*[Kk]cal', dish_line_context):
                continue
            # If it's a price in the direct context of the dish, high confidence
            return price, 85
    
    # If we're still here, look for prices in nearby dishes of the same category
    # Sort all prices by distance from dish name
    sorted_prices = sorted(prices_with_positions, key=lambda x: abs(x[1] - dish_pos))
    
    # Return the closest price that's likely not nutritional info
    for price, _ in sorted_prices:
        return price, 60
    
    return None, 0

# Analyze and select the best document
all_docs = results['documents'][0]
scored_docs = [(doc, score_document_for_dishes(doc, dish_names)) for doc in all_docs]

# Sort by score and select the best one
scored_docs.sort(key=lambda x: x[1], reverse=True)

print("\nRetrieved documents with scores:")
for i, (doc, score) in enumerate(scored_docs):
    print(f"\nDocument {i+1} (Score: {score}):")
    print(doc)

# Use the document with the highest score
data = scored_docs[0][0]
print("\nSelected document:")
print(data)

# Only extract price information if the query is asking about price
extracted_price = None
if is_price_query and dish_names:
    top_dish_name = dish_names[0]
    print("\nDetected price query, extracting price information...")
    
    # Extract price information from relevant documents
    price_info = []
    for doc, score in scored_docs[:3]:  # Check top 3 documents
        if score > 0:
            price, confidence = extract_price(doc, top_dish_name)
            if price:
                price_info.append((price, confidence, doc))

    # Sort price info by confidence
    price_info.sort(key=lambda x: x[1], reverse=True)

    # Get price and document with highest confidence
    if price_info:
        extracted_price, confidence, _ = price_info[0]
        print(f"Extracted price: {extracted_price} rupees (confidence: {confidence})")

print("-----")

# Customize the prompt based on query type
menu_format_guide = """
Menu interpretation tips:
- Prices are typically 3-4 digit numbers (like 1050, 2550, 800) representing cost in rupees
- Nutritional information appears in parentheses like (246 gms, 468 Kcal)
- Dietary indicators appear in parentheses like (n,f,d) where:
  d = dairy, n = nuts, f = fish, g = gluten, etc.
"""

# Build specific guidance based on query type
specific_guidance = ""
if is_price_query:
    specific_guidance += """
When looking for price information:
- Focus on standalone 3-4 digit numbers like 1050, 2550, 800
- Do NOT confuse nutritional information (like "252 gms, 497 Kcal") with prices
- If you find price information, state it clearly as "The price is X rupees"
"""
    if extracted_price:
        # Verify the extracted price is not likely nutritional info by checking the string
        if re.search(fr'[^0-9]{extracted_price}\s*(gms|g|ml|kcal|cal)', data.lower()):
            print("WARNING: Extracted price may be nutritional info - not including in prompt")
        else:
            specific_guidance += f"- Our system identified the price as {extracted_price} rupees\n"
        
if is_ingredient_query:
    specific_guidance += """
When looking for ingredient information:
- Look for descriptive text after the dish name
- Ingredients are often listed after the dish name and price
- List all ingredients you can identify
"""

if is_nutrition_query:
    specific_guidance += """
When looking for nutritional information:
- Look for patterns like (246 gms, 468 Kcal) which indicate weight and calories
- Nutritional info is typically in parentheses
"""

# Generate the final prompt
prompt = f"""
You are a helpful assistant analyzing restaurant menu data.

Here is the menu excerpt:
```
{data}
```

{menu_format_guide if any([is_price_query, is_ingredient_query, is_nutrition_query]) else ""}
{specific_guidance}

Please analyze the menu carefully and answer this question: "{input}"

If the information isn't available in the provided text, say so clearly.
"""

output = ollama.generate(
  model="gemma3:4b",
  prompt=prompt
)

print(output['response'])