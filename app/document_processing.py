from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from fastapi import HTTPException
import os

def load_and_chunk_documents(file_path, chunk_method):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    absolute_path = os.path.abspath(file_path)
    if not os.path.exists(absolute_path):
        raise HTTPException(status_code=400, detail=f"File not found: {absolute_path}")
    loader = TextLoader(absolute_path)
    documents = loader.load()

    if not documents or len(documents) == 0:
        raise HTTPException(status_code=400, detail="No documents were loaded from the provided file path.")

    if chunk_method == 1:
        chunker = SemanticChunker(embeddings, breakpoint_threshold_type="interquartile")
        chunks = chunker.create_documents([doc.page_content for doc in documents])
    elif chunk_method == 2:
        chunker = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=600,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False
        )
        chunks = chunker.split_documents(documents)
    else:
        raise ValueError("Invalid chunk method provided.")
    
    if len(chunks) == 0:
        raise HTTPException(status_code=400, detail="No chunks were created from the document.")

    for chunk in chunks:
        chunk.metadata = {'source': documents[0].metadata.get('source', 'unknown')}
    return chunks

def detect_language(document_content: str) -> str:
    detection_prompt = (
        "You are a language detection assistant. Analyze the following text and respond with the primary language of the text. "
        "Respond only with the language name in English, without explanations."
    )
    text_snippet = document_content[:500]
    messages = [
        {"role": "system", "content": detection_prompt},
        {"role": "user", "content": text_snippet}
    ]
    language_detection_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    response = language_detection_llm.invoke(messages)
    return response.content.strip()
