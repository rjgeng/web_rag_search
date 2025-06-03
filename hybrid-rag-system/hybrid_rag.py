import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from duckduckgo_search import DDGS
import logging
from typing import Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridRAGSystem:
    def __init__(self, documents_folder: str = "../personal-rag-system/me", min_context_length: int = 300):
        """Initialize Hybrid RAG System
        
        Args:
            documents_folder: Path to personal documents folder
            min_context_length: Minimum context length before triggering web search
        """
        self.min_context_length = min_context_length
        self.documents_folder = Path(__file__).parent.absolute() / documents_folder
        
        # Load API key - look for .env in project root (two levels up)
        project_root = Path(__file__).parent.parent
        load_dotenv(project_root / ".env")
        # Load API key - same pattern as other working systems
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=self.api_key)
        self._setup_local_rag()
    
    def _setup_local_rag(self):
        """Setup ChromaDB for local document retrieval"""
        try:
            self.chroma_client = chromadb.Client()
            self.embed_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.api_key,
                model_name="text-embedding-3-small"
            )
            
            try:
                self.collection = self.chroma_client.get_collection("hybrid_personal_profile")
            except:
                self.collection = self.chroma_client.create_collection(
                    "hybrid_personal_profile", 
                    embedding_function=self.embed_fn
                )
                self._load_personal_documents()
            
            logger.info("Local RAG system initialized")
        except Exception as e:
            logger.error(f"Failed to setup local RAG: {e}")
            self.collection = None
    
    def _load_personal_documents(self):
        """Load personal documents into the collection"""
        if not self.collection:
            return
        
        logger.info(f"Loading documents from: {self.documents_folder}")
        
        if not self.documents_folder.exists():
            logger.error(f"Documents folder not found: {self.documents_folder}")
            return
        
        documents = []
        metadatas = []
        ids = []
        
        # Load summary.txt
        summary_file = self.documents_folder / "summary.txt"
        if summary_file.exists():
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary_text = f.read().strip()
                
                if summary_text:
                    documents.append(summary_text)
                    metadatas.append({"source": "summary", "type": "personal"})
                    ids.append("summary_1")
                    logger.info(f"Loaded summary.txt: {len(summary_text)} characters")
                else:
                    logger.warning("summary.txt is empty")
            except Exception as e:
                logger.error(f"Failed to load summary.txt: {e}")
        else:
            logger.warning(f"summary.txt not found at: {summary_file}")
        
        # TODO: Add PDF processing similar to Personal RAG system if needed
        
        if documents:
            try:
                self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
                logger.info(f"Successfully loaded {len(documents)} documents into hybrid collection")
            except Exception as e:
                logger.error(f"Failed to add documents to collection: {e}")
        else:
            logger.warning("No documents were loaded into hybrid collection")
    
    def web_search(self, query: str) -> str:
        """Web search using DuckDuckGo - based on your core logic"""
        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, region='wt-wt', safesearch='Moderate', max_results=3):
                    results.append(f"{r['title']}\n{r['href']}\n{r['body']}\n")
            return "\n".join(results) if results else "No results found."
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return "Web search unavailable."
    
    def get_context_from_local(self, query: str) -> str:
        """Local retrieval from personal documents - based on your core logic"""
        if not self.collection:
            return ""
        
        try:
            results = self.collection.query(query_texts=[query], n_results=3)
            context = "\n".join(sum(results["documents"], []))
            return context
        except Exception as e:
            logger.error(f"Local retrieval failed: {e}")
            return ""
    
    def hybrid_query(self, user_query: str) -> Tuple[str, dict]:
        """Main hybrid query method - implements your core logic"""
        logger.info(f"Processing hybrid query: {user_query}")
        
        # Step 1: Local retrieval
        local_context = self.get_context_from_local(user_query)
        logger.info(f"Local context length: {len(local_context)} characters")
        
        # Step 2: If local context too short, augment with web search
        web_context = ""
        sources_used = ["local"]
        
        if len(local_context) < self.min_context_length:
            logger.info(f"Local context below threshold ({self.min_context_length}), adding web search")
            web_context = self.web_search(user_query)
            sources_used.append("web")
            full_context = f"Local Docs:\n{local_context}\n\nWeb Search:\n{web_context}"
        else:
            logger.info("Local context sufficient, using local only")
            full_context = f"Local Docs:\n{local_context}"
        
        # Step 3: Ask LLM - using your exact prompt structure
        prompt = f"""Answer the following using the provided information ONLY.

{full_context}

Question: {user_query}
Answer:"""
        
        try:
            llm_response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Using mini for cost efficiency
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.1
            )
            
            answer = llm_response.choices[0].message.content
            
            # Return answer and metadata
            metadata = {
                "sources_used": sources_used,
                "local_context_length": len(local_context),
                "web_context_length": len(web_context),
                "threshold_triggered": len(local_context) < self.min_context_length
            }
            
            return answer, metadata
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error generating response: {str(e)}", {"sources_used": [], "error": str(e)}
    
    def query(self, user_query: str) -> str:
        """Simple query interface that returns just the answer"""
        answer, _ = self.hybrid_query(user_query)
        return answer

def print_separator(char="=", length=80):
    """Print a separator line"""
    print(char * length)

def print_query_result(query: str, answer: str, metadata: dict):
    """Pretty print query results with metadata"""
    print(f"🤔 Question: {query}")
    print("─" * 60)
    print(f"💬 Answer: {answer}")
    
    sources = metadata.get("sources_used", [])
    if sources:
        print(f"📚 Sources: {', '.join(sources)}")
    
    if metadata.get("threshold_triggered"):
        print(f"🔍 Hybrid Mode: Local context ({metadata.get('local_context_length', 0)} chars) + Web search")
    else:
        print(f"🏠 Local Only: Sufficient context ({metadata.get('local_context_length', 0)} chars)")
    
    print()
    print_separator()
    print()

def main():
    """Demonstrate the hybrid RAG system"""
    print_separator()
    print("🔄 Hybrid RAG System - Local + Web Search")
    print_separator()
    print()
    
    try:
        # Initialize system with higher threshold to demonstrate web search
        print("📂 Initializing Hybrid RAG System...")
        hybrid_system = HybridRAGSystem(min_context_length=500)  # Increased from 300
        print("✅ System initialized successfully!")
        print()
        
        # Example queries to demonstrate hybrid behavior
        sample_queries = [
            "Tell me about your background and where you're from",  # Should use local only
            "Who is the CEO of OpenAI and what is their background?",  # Should trigger web search
            "What are your food preferences?",  # Should use local only
            "What are the latest developments in AI in 2025?",  # Should trigger web search
            "What is your professional experience?",  # Depends on local content
        ]
        
        print("🎯 Running Sample Queries...")
        print_separator()
        print()
        
        for query in sample_queries:
            answer, metadata = hybrid_system.hybrid_query(query)
            print_query_result(query, answer, metadata)
        
        # Interactive mode
        print("🎮 Interactive Mode - Ask anything!")
        print("(Type 'quit', 'exit', or 'q' to exit)")
        print()
        
        while True:
            try:
                user_query = input("Your question: ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'q', '']:
                    print("\n👋 Thanks for using the Hybrid RAG System!")
                    break
                
                print("\n🔍 Processing your question...\n")
                answer, metadata = hybrid_system.hybrid_query(user_query)
                
                print(f"💬 Answer: {answer}")
                sources = metadata.get("sources_used", [])
                if sources:
                    print(f"📚 Sources: {', '.join(sources)}")
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Thanks for using the Hybrid RAG System!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print()
    
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        print(f"\n❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure you have run the personal-rag-system first to create documents")
        print("2. Check that the .env file exists in the project root with OPENAI_API_KEY")
        print("3. Verify all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()