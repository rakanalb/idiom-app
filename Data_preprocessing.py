import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
import re
from tqdm import tqdm
import faiss
import numpy as np
import pickle  # Import pickle for serialization

def extract_text_from_pdf(pdf_path, start_page=6, end_page_offset=6):
    """Extracts text from a PDF, skipping the first and last few pages."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        end_page = len(doc) - end_page_offset
        
        for page_num in tqdm(range(start_page, end_page), desc="Extracting text"):
            page_text = doc[page_num].get_text()
            page_text = re.sub(r'[^\x00-\x7F]+', '', page_text)
            text += page_text
            
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return None

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Splits text into chunks using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.create_documents([text])

def create_embeddings(docs):
    """Creates embeddings for each document chunk."""
    embeddings = OpenAIEmbeddings()
    embedded_docs = []
    for doc in tqdm(docs, desc="Embedding chunks"):
        try:
            embedding = embeddings.embed_query(doc.page_content)
            embedded_docs.append(embedding)
        except Exception as e:
            print(f"Error embedding chunk: {str(e)}")
            continue
    return embedded_docs

def create_faiss_index(embedded_docs):
    """Creates a FAISS index from the embedded documents."""
    embedding_matrix = np.array(embedded_docs).astype('float32')
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)
    faiss.write_index(index, "faiss_index.idx")
    print("FAISS index created and saved.")
    return index

def display_vectors(index, num_vectors=5):
    """Displays a specified number of vectors from the FAISS index."""
    vectors = index.reconstruct_n(0, num_vectors)
    for i, vector in enumerate(vectors):
        print(f"Vector {i}: {vector}")

def save_vectors(embedded_docs, filename="vectors.pkl"):
    """Saves the embedded documents to a file using pickle."""
    with open(filename, 'wb') as f:
        pickle.dump(embedded_docs, f)
    print(f"Vectors saved to {filename}")

def main():
    pdf_path = "idioms.pdf"
    text = extract_text_from_pdf(pdf_path)
    if text is None:
        raise Exception("Failed to extract text from PDF")
    
    docs = split_text_into_chunks(text)
    embedded_docs = create_embeddings(docs)
    print(f"Successfully processed {len(embedded_docs)} chunks")
    
    index = create_faiss_index(embedded_docs)
    display_vectors(index)
    
    # Save the vectors using pickle
    save_vectors(embedded_docs)

if __name__ == "__main__":
    main()

