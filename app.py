import os
import tempfile

from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from llama_cpp import Llama
from prompts import generate_prompt
from langchain.globals import set_debug, get_debug
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

os.environ["HF_HUB_OFFLINE"] = "1"  # Critical - prevents online checks
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Critical - prevents tokenization issues

# Initialize Flask
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'   # sets a folder to save uploaded files
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load llama model only once
llm = None  # Global model object
MODEL_PATH = "./models/capybarahermes-2.5-mistral-7b.Q4_0.gguf"

def load_model():
    global llm
    if llm is None:
        llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_threads=8, n_gpu_layers=40, n_batch=512, use_mmap=True, use_mlock=False)

# Save uploaded file
def save_uploaded_file(uploaded_file):
    content = uploaded_file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        return tmp.name

# PDF parsing
def load_and_split_pdf(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()  
        splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n"
        )
        return splitter.split_documents(pages) if pages else None
    except Exception as e:
        print(f"PDF Loading Error: {e}")
        return None
    

# Build vector store
def build_vector_store(documents):
    '''model = SentenceTransformer(
        './models/embeddings/all-MiniLM-L6-v2',  # Local path
        device="cpu"
    )
    class LocalEmbeddings:
        def embed_documents(self, texts):
            return model.encode(texts, show_progress_bar=False)
    
    embeddings = LocalEmbeddings()
    return Chroma.from_documents(
        documents, 
        embedding=embeddings,
        persist_directory="./chroma_temp"
    )'''
    embeddings = HuggingFaceEmbeddings(model_name = "all-minilm-l6-v2", model_kwargs={"device": "cpu"})
    return Chroma.from_documents(documents, embedding=embeddings, persist_directory="./chroma_temp", collection_metadata={"hnsw:space" : "cosine"})

# Retrieve relevant chunks
def get_context_from_query(vector_store, query):
    retriever = vector_store.as_retriever(search_type = "mmr",search_kwargs={"k": 4, "score_threshold": 0.7})
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content[:500] for doc in docs])

# Clean output to remove template markers
def clean_output(text):
    return text.split("<<ANSWER>>")[-1].strip()

# Ask Llama model
def ask_llamacpp(context, question):
    load_model()  # Ensure model is loaded
    prompt = generate_prompt(context, question)
    output = llm(prompt, max_tokens=512, temperature=0.7, top_p=0.9, stop=["</s>","###"])
    return clean_output(output["choices"][0]["text"])

# Serve frontend
@app.route('/')
def index():
    return render_template('frontend2.html')

# Handle chat POST
@app.route('/chat', methods=['POST'])
def chat():
    if 'file' not in request.files or not request.form.get('question'):
        return jsonify({"error": "File and question are required"}), 400

    temp_path = None
    try:
        # Create temp file with unique name
        temp_path = os.path.join(tempfile.gettempdir(), f"pdf_{os.urandom(4).hex()}.pdf")
        request.files['file'].save(temp_path)
        
        # Process PDF (explicitly close resources)
        documents = load_and_split_pdf(temp_path)
        if not documents:
            return jsonify({"error": "Unreadable PDF content"}), 400
            
        vector_store = build_vector_store(documents)
        context = get_context_from_query(vector_store, request.form['question'])
        answer = ask_llamacpp(context, request.form['question'])
        
        return jsonify({
            "answer": answer,
            "question": request.form['question']
        })
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        # Ensure file cleanup
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except PermissionError:
                # If still locked, schedule deletion on next reboot
                import ctypes
                ctypes.windll.kernel32.MoveFileExW(temp_path, None, 4)
    
# Run the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
