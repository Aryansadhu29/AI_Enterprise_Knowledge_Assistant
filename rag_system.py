"""
Enterprise Knowledge Assistant - RAG System with LLM
Uses Groq API, Ollama, or Hugging Face
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


class DocumentProcessor:
    """Handles document loading and chunking"""
    
    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        """
        Initialize document processor
        
        Args:
            chunk_size: Characters per chunk
            overlap: Characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def load_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return ""
    
    def load_text(self, txt_path: str) -> str:
        """Load text from plain text file"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading text file {txt_path}: {e}")
            return ""
    
    def load_document(self, file_path: str) -> Tuple[str, str]:
        """
        Load document from file
        
        Returns:
            Tuple of (text_content, file_name)
        """
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.pdf':
            content = self.load_pdf(str(file_path))
        elif file_path.suffix.lower() in ['.txt', '.md']:
            content = self.load_text(str(file_path))
        else:
            print(f"Unsupported file type: {file_path.suffix}")
            return "", ""
        
        return content, file_path.name
    
    def chunk_text(self, text: str, doc_name: str) -> List[Dict]:
        """
        Split text into overlapping chunks
        
        Returns:
            List of chunk dictionaries with content and metadata
        """
        chunks = []
        text = text.strip()
        
        if not text:
            return chunks
        
        # Split by paragraphs first, then by character limit
        paragraphs = text.split('\n\n')
        current_chunk = ""
        chunk_id = 0
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < self.chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                # Save current chunk
                if current_chunk.strip():
                    chunks.append({
                        'id': f"{doc_name}_chunk_{chunk_id}",
                        'content': current_chunk.strip(),
                        'document': doc_name,
                        'chunk_num': chunk_id
                    })
                    chunk_id += 1
                
                # Start new chunk with overlap
                current_chunk = current_chunk[-self.overlap:] + paragraph + "\n\n"
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append({
                'id': f"{doc_name}_chunk_{chunk_id}",
                'content': current_chunk.strip(),
                'document': doc_name,
                'chunk_num': chunk_id
            })
        
        print(f"Created {len(chunks)} chunks from {doc_name}")
        return chunks


class EmbeddingManager:
    """Handles embedding generation and storage"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding model (Runs locally)
        
        Args:
            model_name: Name of sentence transformer model
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a text string"""
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)


class VectorDatabase:
    """Handles FAISS vector database operations"""
    
    def __init__(self, embedding_dim: int):
        """
        Initialize FAISS vector database (Runs locally)
        
        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.chunks = []
        self.id_to_chunk = {}
    
    def add_chunks(self, chunks: List[Dict], embeddings: np.ndarray) -> None:
        """Add chunks and their embeddings to the database"""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        embeddings = embeddings.astype('float32')
        self.index.add(embeddings)
        
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.id_to_chunk[len(self.chunks) - 1] = chunk
        
        print(f"Added {len(chunks)} chunks to vector database")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Search for most similar chunks
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
        
        Returns:
            List of similar chunks with distances
        """
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['distance'] = float(distance)
                chunk['similarity_score'] = 1 / (1 + float(distance))
                results.append(chunk)
        
        return results
    
    def save(self, directory: str) -> None:
        """Save index and metadata to disk"""
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, 'faiss.index'))
        
        with open(os.path.join(directory, 'chunks.json'), 'w') as f:
            json.dump(self.chunks, f, indent=2)
        
        print(f"Saved vector database to {directory}")
    
    def load(self, directory: str) -> None:
        """Load index and metadata from disk"""
        self.index = faiss.read_index(os.path.join(directory, 'faiss.index'))
        
        with open(os.path.join(directory, 'chunks.json'), 'r') as f:
            self.chunks = json.load(f)
        
        for i, chunk in enumerate(self.chunks):
            self.id_to_chunk[i] = chunk
        
        print(f"Loaded vector database from {directory}")


class FreeLLMAnswerGenerator:
    """Generates answers using LLMs"""
    
    def __init__(self, llm_type: str = 'groq'):
        """
        Initialize LLM
        
        Args:
            llm_type: 'groq' (online), 'ollama' (local), 'huggingface' (online)
        """
        self.llm_type = llm_type
        
        if llm_type == 'groq':
            self._init_groq()
        elif llm_type == 'ollama':
            self._init_ollama()
        elif llm_type == 'huggingface':
            self._init_huggingface()
        else:
            print(f"Unknown LLM type: {llm_type}, using simple fallback")
    
    def _init_groq(self):
        """Initialize Groq"""
        try:
            import groq
            api_key = os.environ.get('GROQ_API_KEY')
            if not api_key:
                print("GROQ_API_KEY not found!")
                print("Get key at: https://console.groq.com/keys")
                print("Then set: export GROQ_API_KEY= Your_API_Key")
                self.client = None
                return
            
            self.client = groq.Groq(api_key=api_key)
            print("Groq LLM initialized")
        except ImportError:
            print("groq not installed. Run: pip install groq")
            self.client = None
    
    def _init_ollama(self):
        """Initialize Ollama (Runs locally, no internet!)"""
        try:
            import ollama
            self.client = ollama
            print("Ollama LLM initialized (Runs locally)")
            print("  Make sure Ollama is running: ollama serve")
        except ImportError:
            print("ollama not installed. Run: pip install ollama")
            self.client = None
    
    def _init_huggingface(self):
        """Initialize Hugging Face"""
        try:
            from transformers import pipeline
            print("Loading Hugging Face model (may take a minute)...")
            self.client = pipeline("text-generation", model="gpt2")
            print("Hugging Face LLM initialized")
        except ImportError:
            print("transformers not installed. Run: pip install transformers torch")
            self.client = None
    
    def generate(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate answer using LLM"""
        
        if not context_chunks:
            return "I don't have information about this in the knowledge base."
        
        # Prepare context
        context_text = "\n\n".join([
            f"From {chunk['document']}:\n{chunk['content']}"
            for chunk in context_chunks[:3]  # Limit to 3 chunks to save tokens
        ])
        
        # Create prompt
        prompt = f"""You are a helpful enterprise assistant. Answer the user's question based ONLY on the provided context.

If the answer is not in the context, say "I don't have information about this in the knowledge base."

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""
        
        try:
            if self.llm_type == 'groq':
                return self._generate_groq(prompt)
            elif self.llm_type == 'ollama':
                return self._generate_ollama(prompt)
            elif self.llm_type == 'huggingface':
                return self._generate_huggingface(prompt)
            else:
                return self._generate_simple(query, context_chunks)
        except Exception as e:
            print(f"Error generating answer: {e}")
            return self._generate_simple(query, context_chunks)
    
    def _generate_groq(self, prompt: str) -> str:
        """Generate using Groq"""
        if not self.client:
            return self._generate_simple_fallback(prompt)
        
        try:
            message = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="mixtral-8x7b-32768",  # FREE model
                max_tokens=500,
                temperature=0.3
            )
            return message.choices[0].message.content
        except Exception as e:
            print(f"Groq error: {e}")
            return self._generate_simple_fallback(prompt)
    
    def _generate_ollama(self, prompt: str) -> str:
        """Generate using Ollama"""
        if not self.client:
            return self._generate_simple_fallback(prompt)
        
        try:
            response = self.client.generate(
                model="mistral",  # Popular free model
                prompt=prompt,
                stream=False
            )
            return response['response']
        except Exception as e:
            print(f"Ollama error (make sure 'ollama serve' is running): {e}")
            return self._generate_simple_fallback(prompt)
    
    def _generate_huggingface(self, prompt: str) -> str:
        """Generate using Hugging Face"""
        if not self.client:
            return self._generate_simple_fallback(prompt)
        
        try:
            # Note: GPT-2 has limited context, may need larger model
            response = self.client(
                prompt,
                max_length=200,
                num_return_sequences=1,
                temperature=0.3
            )
            return response[0]['generated_text'][len(prompt):]
        except Exception as e:
            print(f"HuggingFace error: {e}")
            return self._generate_simple_fallback(prompt)
    
    def _generate_simple_fallback(self, prompt: str) -> str:
        """Simple fallback - return relevant chunk"""
        # Extract the question from prompt
        if "QUESTION:" in prompt:
            return prompt.split("CONTEXT:")[-1][:500]
        return "Unable to generate answer. Please check your LLM setup."


class RAGSystem:
    """Complete Retrieval Augmented Generation system with LLMs"""
    
    def __init__(self, db_path: str = './vector_db', llm_type: str = 'groq'):
        """
        Initialize RAG system with LLM
        
        Args:
            db_path: Path to save vector database
            llm_type: 'groq', 'ollama', 'huggingface'
        """
        print(f"\n{'='*60}")
        print(f"Initializing RAG System with {llm_type.upper()}")
        print(f"{'='*60}\n")
        
        self.db_path = db_path
        self.llm_type = llm_type
        self.doc_processor = DocumentProcessor(chunk_size=800, overlap=100)
        self.embeddings = EmbeddingManager('all-MiniLM-L6-v2')
        self.vector_db = VectorDatabase(self.embeddings.embedding_dim)
        self.llm = FreeLLMAnswerGenerator(llm_type)
        self.loaded_documents = []
    
    def ingest_documents(self, document_paths: List[str]) -> None:
        """
        Process and ingest documents into the system
        
        Args:
            document_paths: List of file paths to load
        """
        all_chunks = []
        all_texts = []
        
        print(f"\nIngesting {len(document_paths)} documents...")
        
        for doc_path in document_paths:
            print(f"\n  Processing: {doc_path}")
            
            # Load document
            content, doc_name = self.doc_processor.load_document(doc_path)
            
            if not content:
                print(f"   Skipping {doc_path} - no content extracted")
                continue
            
            # Chunk the document
            chunks = self.doc_processor.chunk_text(content, doc_name)
            all_chunks.extend(chunks)
            all_texts.extend([chunk['content'] for chunk in chunks])
            self.loaded_documents.append(doc_name)
        
        if not all_chunks:
            print(" No chunks created from documents!")
            return
        
        # Generate embeddings
        print(f"\nGenerating embeddings for {len(all_chunks)} chunks...")
        embeddings = self.embeddings.embed_batch(all_texts)
        
        # Add to vector database
        self.vector_db.add_chunks(all_chunks, embeddings)
        
        # Save the database
        self.vector_db.save(self.db_path)
        
        print(f"\nSuccessfully ingested {len(self.loaded_documents)} documents")
        print(f"Created {len(all_chunks)} chunks")
    
    def retrieve_context(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: User question
            k: Number of chunks to retrieve
        
        Returns:
            List of relevant chunks with scores
        """
        query_embedding = self.embeddings.embed_text(query)
        results = self.vector_db.search(query_embedding, k=k)
        return results
    
    def answer_question(self, query: str, k: int = 5) -> Dict:
        """
        Answer a user question using the RAG system (with LLM)
        
        Args:
            query: User question
            k: Number of context chunks to retrieve
        
        Returns:
            Dictionary with answer and sources
        """
        print(f"\nQuestion: {query}")
        
        # Retrieve relevant chunks
        context_chunks = self.retrieve_context(query, k=k)
        
        if not context_chunks:
            return {
                'answer': "I don't have information about this in the knowledge base.",
                'sources': [],
                'query': query,
                'retrieved_chunks': 0
            }
        
        print(f"✓ Retrieved {len(context_chunks)} relevant chunks")
        
        # Generate answer using FREE LLM
        answer = self.llm.generate(query, context_chunks)
        
        # Prepare sources
        sources = []
        for chunk in context_chunks:
            source = {
                'document': chunk['document'],
                'chunk': chunk['chunk_num'],
                'similarity': chunk['similarity_score']
            }
            if source not in sources:
                sources.append(source)
        
        return {
            'answer': answer,
            'sources': sources,
            'query': query,
            'retrieved_chunks': len(context_chunks)
        }
    
    def load_from_disk(self, db_path: str = None) -> bool:
        """Load previously saved vector database"""
        if db_path is None:
            db_path = self.db_path
        
        if not os.path.exists(os.path.join(db_path, 'faiss.index')):
            print(f"No saved database found at {db_path}")
            return False
        
        try:
            self.vector_db.load(db_path)
            print(f"Loaded vector database from {db_path}")
            return True
        except Exception as e:
            print(f"Error loading database: {e}")
            return False


def create_sample_documents():
    """Create sample documents for testing"""
    os.makedirs('documents', exist_ok=True)
    
    # HR Policy
    hr_policy = """
HR POLICY DOCUMENT

1. LEAVE POLICY
Employees are eligible for the following annual leaves:
- Casual Leave: 12 days per year
- Paid Leave: 20 days per year
- Sick Leave: 10 days per year
- Maternity Leave: 6 months with pay
- Paternity Leave: 10 days with pay

Employees can carry forward up to 5 unused casual leaves to the next year.
Paid leaves cannot be carried forward and must be taken within the year.

2. WORK HOURS
Regular working hours are from 9:00 AM to 6:00 PM, Monday to Friday.
Employees get a 1-hour lunch break.
Flexible work arrangements are available upon manager approval.

3. REMOTE WORK POLICY
Employees can work from home up to 2 days per week.
Prior approval from manager is required.
All company equipment must be returned when employee leaves.

4. PERFORMANCE REVIEW
Performance reviews are conducted annually in December.
Promotions are based on performance ratings and availability of positions.
Salary increments are typically 5-15% based on performance.
"""
    
    with open('documents/HR_Policy.txt', 'w') as f:
        f.write(hr_policy)
    
    # Product Guide
    product_guide = """
PRODUCT GUIDE - ACME SUITE

Overview:
ACME Suite is our enterprise software solution designed for businesses.

1. FEATURES
- Cloud-based storage: 100GB per user
- Real-time collaboration tools
- Advanced analytics dashboard
- API access for integrations
- 24/7 customer support

2. PRICING
Basic Plan: $99/month (up to 5 users)
Professional Plan: $299/month (up to 50 users)
Enterprise Plan: Custom pricing (unlimited users)

3. IMPLEMENTATION
Our implementation takes 2-4 weeks depending on complexity.
We provide free training for the first 50 users.
Migration from legacy systems is supported.

4. REFUND POLICY
Refunds are allowed within 30 days of purchase if the service doesn't meet requirements.
After 30 days, refunds are not available but users can request service credits.
Setup fees are non-refundable.

5. SUPPORT
Standard support: Email (24-hour response)
Premium support: Phone + Email (1-hour response)
Enterprise support: Dedicated account manager
"""
    
    with open('documents/Product_Guide.txt', 'w') as f:
        f.write(product_guide)
    
    # Compliance Guide
    compliance_guide = """
COMPLIANCE AND GOVERNANCE GUIDELINES

1. DATA PROTECTION
All employee data must be encrypted at rest and in transit.
Personal data should not be shared without explicit consent.
Regular data audits are conducted quarterly.
GDPR compliance is mandatory for all European operations.

2. SECURITY REQUIREMENTS
Passwords must be at least 12 characters with mixed case.
Two-factor authentication is required for all admin accounts.
VPN usage is mandatory when accessing company systems remotely.
All devices must have anti-malware software installed.

3. INCIDENT RESPONSE
Security incidents must be reported within 24 hours.
A incident response team will investigate within 48 hours.
Affected users are notified within 72 hours of a breach.

4. AUDIT REQUIREMENTS
Annual security audits are mandatory.
Compliance certifications: ISO 27001, SOC 2 Type II
Third-party penetration testing is conducted annually.

5. ACCEPTABLE USE POLICY
Company systems should be used for business purposes only.
Personal use is limited to break times.
Downloading unauthorized software is prohibited.
Violations can result in disciplinary action up to termination.
"""
    
    with open('documents/Compliance_Guide.txt', 'w') as f:
        f.write(compliance_guide)
    
    print("Sample documents created in 'documents/' directory")


def main():
    """Example usage with LLM"""
    
    # Choose your FREE LLM (uncomment one):
    llm_type = 'groq'  # Free online (10K requests/month free)
    # llm_type = 'ollama'  # Free local (requires Ollama installed)
    # llm_type = 'huggingface'  # Free local (requires transformers installed)
    
    # Create RAG system with FREE LLM
    rag = RAGSystem(db_path='./vector_db', llm_type=llm_type)
    
    # Check if we have a saved database
    if not rag.load_from_disk():
        # If not, ingest documents
        create_sample_documents()
        
        documents = [
            'documents/HR_Policy.txt',
            'documents/Product_Guide.txt',
            'documents/Compliance_Guide.txt'
        ]
        
        rag.ingest_documents(documents)
    
    # Ask questions
    questions = [
        "What is the employee leave policy?",
        "How many paid leaves are available?",
        "What is the refund policy?",
        "Tell me about compliance requirements",
    ]
    
    for question in questions:
        result = rag.answer_question(question)
        print("\n" + "="*80)
        print(f"Q: {result['query']}")
        print(f"\nA: {result['answer']}")
        print(f"\nSources ({len(result['sources'])}):")
        for source in result['sources']:
            print(f"  - {source['document']} (Chunk {source['chunk']}, Similarity: {source['similarity']:.2%})")
        print("="*80)


if __name__ == "__main__":
    main()
