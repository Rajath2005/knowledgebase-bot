import os
import json
import logging
import requests
import numpy as np
import re
import sqlite3
import hashlib
import time
from datetime import datetime, timedelta
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import pickle
import pandas as pd
from typing import List, Dict, Tuple, Optional
import uuid

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create Flask app with enhanced configuration
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'knowledge_uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Multiple AI API configurations with free options
AI_APIS = {
    'huggingface': {
        'url': "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium",
        'key': os.getenv("HUGGINGFACE_API_KEY", ""),
        'headers_template': {"Authorization": "Bearer {key}", "Content-Type": "application/json"},
        'payload_template': {"inputs": "{message}", "options": {"wait_for_model": True}},
        'free': False
    },
    'gemini': {
        'url': "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
        'key': os.getenv("GEMINI_API_KEY", ""),
        'headers_template': {"Content-Type": "application/json"},
        'payload_template': {
            "contents": [{"parts": [{"text": "{message}"}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1000}
        },
        'free': True,  # Has generous free tier
        'url_with_key': True
    },
    'deepseek': {
        'url': "https://api.deepseek.com/v1/chat/completions",
        'key': os.getenv("DEEPSEEK_API_KEY", ""),
        'headers_template': {"Authorization": "Bearer {key}", "Content-Type": "application/json"},
        'payload_template': {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "{message}"}],
            "temperature": 0.7
        },
        'free': True  # Has free tier
    },
    'ollama': {
        'url': "http://localhost:11434/api/generate",
        'key': "",  # No key needed for local Ollama
        'headers_template': {"Content-Type": "application/json"},
        'payload_template': {
            "model": "llama2",
            "prompt": "{message}",
            "stream": False
        },
        'free': True,  # Completely free local deployment
        'local': True
    },
    'openrouter': {
        'url': "https://openrouter.ai/api/v1/chat/completions",
        'key': os.getenv("OPENROUTER_API_KEY", ""),
        'headers_template': {"Authorization": "Bearer {key}", "Content-Type": "application/json"},
        'payload_template': {
            "model": "microsoft/DialoGPT-medium",
            "messages": [{"role": "user", "content": "{message}"}]
        },
        'free': True  # Has free models available
    }
}

# Enhanced Knowledge Base Management
class KnowledgeBaseManager:
    def __init__(self, db_path='knowledge_base.db'):
        self.db_path = db_path
        self.vectorizer = None
        self.document_embeddings = None
        self.lsa_model = None  # Latent Semantic Analysis for better understanding
        self.documents = []
        self.document_metadata = []
        self.init_database()
        self.load_knowledge_base()
    
    def init_database(self):
        """Initialize SQLite database for persistent knowledge storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT,
                content TEXT,
                category TEXT,
                source TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                embedding_hash TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_message TEXT,
                bot_response TEXT,
                api_used TEXT,
                knowledge_used TEXT,
                timestamp TIMESTAMP,
                feedback_score INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_data (
                id TEXT PRIMARY KEY,
                input_text TEXT,
                expected_output TEXT,
                category TEXT,
                created_at TIMESTAMP,
                is_validated BOOLEAN
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_document(self, title: str, content: str, category: str = "general", 
                    source: str = "manual", metadata: Dict = None) -> str:
        """Add a new document to the knowledge base"""
        doc_id = str(uuid.uuid4())
        now = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO documents (id, title, content, category, source, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (doc_id, title, content, category, source, now, now, json.dumps(metadata or {})))
        
        conn.commit()
        conn.close()
        
        # Rebuild embeddings
        self.load_knowledge_base()
        
        logger.info(f"Added document: {title} (ID: {doc_id})")
        return doc_id
    
    def add_training_data(self, input_text: str, expected_output: str, category: str = "general"):
        """Add training data for fine-tuning responses"""
        training_id = str(uuid.uuid4())
        now = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO training_data (id, input_text, expected_output, category, created_at, is_validated)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (training_id, input_text, expected_output, category, now, False))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Added training data: {input_text[:50]}...")
        return training_id
    
    def load_knowledge_base(self):
        """Load all documents and create embeddings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT title, content, category, source, metadata FROM documents')
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            # Add default knowledge if database is empty
            self._add_default_knowledge()
            return self.load_knowledge_base()
        
        self.documents = []
        self.document_metadata = []
        
        for title, content, category, source, metadata_json in rows:
            self.documents.append(content)
            self.document_metadata.append({
                'title': title,
                'category': category,
                'source': source,
                'metadata': json.loads(metadata_json or '{}')
            })
        
        # Create enhanced embeddings with LSA
        self._create_embeddings()
        
        logger.info(f"Loaded {len(self.documents)} documents into knowledge base")
    
    def _add_default_knowledge(self):
        """Add default knowledge base"""
        default_docs = [
            {
                'title': 'Python Programming Basics',
                'content': '''Python is a high-level, interpreted programming language created by Guido van Rossum. 
                It's known for its clean syntax, readability, and extensive libraries. Python supports multiple 
                programming paradigms including object-oriented, functional, and procedural programming. 
                Common uses include web development (Django, Flask), data science (Pandas, NumPy), 
                machine learning (TensorFlow, PyTorch), automation, and scientific computing.''',
                'category': 'programming'
            },
            {
                'title': 'Machine Learning Fundamentals',
                'content': '''Machine Learning is a subset of AI that enables computers to learn patterns 
                from data without explicit programming. Main types include supervised learning (using labeled data), 
                unsupervised learning (finding patterns in unlabeled data), and reinforcement learning 
                (learning through rewards). Common algorithms include linear regression, decision trees, 
                neural networks, and clustering. Applications span recommendation systems, image recognition, 
                natural language processing, and predictive analytics.''',
                'category': 'ai'
            },
            {
                'title': 'Web Development with Flask',
                'content': '''Flask is a lightweight Python web framework that follows the WSGI standard. 
                It provides essential tools for web development while maintaining flexibility and simplicity. 
                Key features include URL routing, template engine (Jinja2), request handling, and database 
                integration through extensions. Flask is ideal for APIs, small to medium applications, 
                and microservices. It offers more control over application architecture compared to 
                full-featured frameworks like Django.''',
                'category': 'web_development'
            }
        ]
        
        for doc in default_docs:
            self.add_document(doc['title'], doc['content'], doc['category'], 'default')
    
    def _create_embeddings(self):
        """Create enhanced TF-IDF embeddings with LSA"""
        if not self.documents:
            return
        
        # Enhanced TF-IDF with better parameters
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams for better context
            min_df=1,
            max_df=0.8,
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{2,}\b',
            sublinear_tf=True  # Use log scaling
        )
        
        # Create TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        
        # Apply LSA for semantic understanding
        self.lsa_model = TruncatedSVD(n_components=min(100, len(self.documents)))
        self.document_embeddings = self.lsa_model.fit_transform(tfidf_matrix)
        
        logger.info(f"Created embeddings with {self.document_embeddings.shape[1]} dimensions")
    
    def find_relevant_documents(self, query: str, top_k: int = 3, threshold: float = 0.1) -> List[Tuple[str, float, Dict]]:
        """Find most relevant documents using enhanced similarity"""
        if not self.vectorizer or not self.document_embeddings.size:
            return []
        
        # Transform query
        query_tfidf = self.vectorizer.transform([query])
        query_lsa = self.lsa_model.transform(query_tfidf)
        
        # Calculate similarities
        similarities = cosine_similarity(query_lsa, self.document_embeddings).flatten()
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > threshold:
                results.append((
                    self.documents[idx],
                    similarities[idx],
                    self.document_metadata[idx]
                ))
        
        return results
    
    def get_training_examples(self, category: str = None) -> List[Dict]:
        """Get training examples for fine-tuning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if category:
            cursor.execute('SELECT input_text, expected_output, category FROM training_data WHERE category = ?', (category,))
        else:
            cursor.execute('SELECT input_text, expected_output, category FROM training_data')
        
        rows = cursor.fetchall()
        conn.close()
        
        return [{'input': row[0], 'output': row[1], 'category': row[2]} for row in rows]

# Initialize knowledge base manager
kb_manager = KnowledgeBaseManager()

class AIAPIManager:
    def __init__(self):
        self.apis = AI_APIS
        self.last_api_call = {}
        self.api_call_counts = {}
        self.rate_limits = {
            'gemini': {'calls_per_minute': 60, 'calls_per_day': 1500},
            'deepseek': {'calls_per_minute': 20, 'calls_per_day': 500},
            'huggingface': {'calls_per_minute': 30, 'calls_per_day': 1000}
        }
    
    def _check_rate_limit(self, api_name: str) -> bool:
        """Check if API call is within rate limits"""
        now = time.time()
        
        # Initialize tracking if needed
        if api_name not in self.api_call_counts:
            self.api_call_counts[api_name] = {'minute': [], 'day': []}
        
        # Clean old entries
        minute_ago = now - 60
        day_ago = now - 86400
        
        self.api_call_counts[api_name]['minute'] = [
            t for t in self.api_call_counts[api_name]['minute'] if t > minute_ago
        ]
        self.api_call_counts[api_name]['day'] = [
            t for t in self.api_call_counts[api_name]['day'] if t > day_ago
        ]
        
        # Check limits
        limits = self.rate_limits.get(api_name, {'calls_per_minute': 100, 'calls_per_day': 10000})
        
        if len(self.api_call_counts[api_name]['minute']) >= limits['calls_per_minute']:
            return False
        if len(self.api_call_counts[api_name]['day']) >= limits['calls_per_day']:
            return False
        
        return True
    
    def _record_api_call(self, api_name: str):
        """Record an API call for rate limiting"""
        now = time.time()
        if api_name not in self.api_call_counts:
            self.api_call_counts[api_name] = {'minute': [], 'day': []}
        
        self.api_call_counts[api_name]['minute'].append(now)
        self.api_call_counts[api_name]['day'].append(now)
    
    def query_api(self, api_name: str, message: str, context: str = None) -> Dict:
        """Query a specific AI API"""
        if api_name not in self.apis:
            return {"success": False, "error": f"API '{api_name}' not configured"}
        
        api_config = self.apis[api_name]
        
        # Check if API key is available (skip for local APIs)
        if not api_config.get('local', False) and not api_config['key']:
            return {"success": False, "error": f"API key not configured for {api_name}"}
        
        # Check rate limits
        if not self._check_rate_limit(api_name):
            return {"success": False, "error": f"Rate limit exceeded for {api_name}"}
        
        try:
            # Prepare headers
            headers = {}
            for key, template in api_config['headers_template'].items():
                if '{key}' in template:
                    headers[key] = template.format(key=api_config['key'])
                else:
                    headers[key] = template
            
            # Prepare payload
            enhanced_message = message
            if context:
                enhanced_message = f"Context: {context}\n\nUser Question: {message}\n\nPlease provide a helpful response based on the context."
            
            payload = self._format_payload(api_config['payload_template'], enhanced_message)
            
            # Prepare URL
            url = api_config['url']
            if api_config.get('url_with_key') and api_config['key']:
                url += f"?key={api_config['key']}"
            
            logger.info(f"Querying {api_name} API...")
            
            # Make request
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                self._record_api_call(api_name)
                
                # Parse response based on API format
                parsed_response = self._parse_api_response(api_name, result)
                
                return {"success": True, "data": parsed_response, "api_used": api_name}
            else:
                error_msg = f"{api_name} API error: {response.status_code} - {response.text[:200]}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg, "status_code": response.status_code}
                
        except requests.exceptions.Timeout:
            return {"success": False, "error": f"Request to {api_name} API timed out"}
        except Exception as e:
            return {"success": False, "error": f"Error calling {api_name} API: {str(e)}"}
    
    def _format_payload(self, template: Dict, message: str) -> Dict:
        """Recursively format payload template"""
        if isinstance(template, dict):
            return {k: self._format_payload(v, message) for k, v in template.items()}
        elif isinstance(template, list):
            return [self._format_payload(item, message) for item in template]
        elif isinstance(template, str):
            return template.format(message=message)
        else:
            return template
    
    def _parse_api_response(self, api_name: str, response: Dict) -> str:
        """Parse API response based on the specific API format"""
        try:
            if api_name == 'gemini':
                return response['candidates'][0]['content']['parts'][0]['text']
            elif api_name == 'deepseek':
                return response['choices'][0]['message']['content']
            elif api_name == 'openrouter':
                return response['choices'][0]['message']['content']
            elif api_name == 'ollama':
                return response['response']
            elif api_name == 'huggingface':
                if isinstance(response, list) and len(response) > 0:
                    return response[0].get('generated_text', str(response[0]))
                return str(response)
            else:
                return str(response)
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing {api_name} response: {e}")
            return f"Error parsing response from {api_name}"
    
    def get_best_available_api(self) -> str:
        """Get the best available API based on configuration and rate limits"""
        # Priority order: free APIs first, then paid
        priority_apis = ['ollama', 'deepseek', 'gemini', 'openrouter', 'huggingface']
        
        for api_name in priority_apis:
            if api_name in self.apis:
                api_config = self.apis[api_name]
                # Check if API is available
                if api_config.get('local', False) or api_config['key']:
                    if self._check_rate_limit(api_name):
                        return api_name
        
        return None

# Initialize API manager
api_manager = AIAPIManager()

# Enhanced response generation
def generate_enhanced_response(user_message: str, api_preference: str = None) -> Dict:
    """Generate response using RAG and multiple AI APIs"""
    
    # Find relevant documents
    relevant_docs = kb_manager.find_relevant_documents(user_message, top_k=2)
    
    # Prepare context
    context = None
    knowledge_used = []
    if relevant_docs:
        context_parts = []
        for doc_content, similarity, metadata in relevant_docs:
            context_parts.append(f"[{metadata['title']}]: {doc_content[:300]}...")
            knowledge_used.append(metadata['title'])
        context = "\n\n".join(context_parts)
    
    # Get best API
    api_to_use = api_preference or api_manager.get_best_available_api()
    
    if not api_to_use:
        # Fallback to knowledge-based response
        if context:
            return {
                "response": f"Based on my knowledge: {relevant_docs[0][0][:500]}...",
                "api_used": "knowledge_base",
                "knowledge_used": knowledge_used,
                "confidence": relevant_docs[0][1]
            }
        else:
            return {
                "response": "I'd be happy to help! Could you provide more specific details about what you're looking for?",
                "api_used": "fallback",
                "knowledge_used": [],
                "confidence": 0.0
            }
    
    # Query AI API
    api_response = api_manager.query_api(api_to_use, user_message, context)
    
    if api_response["success"]:
        return {
            "response": api_response["data"],
            "api_used": api_to_use,
            "knowledge_used": knowledge_used,
            "confidence": relevant_docs[0][1] if relevant_docs else 0.0
        }
    else:
        # Try fallback API
        fallback_apis = ['ollama', 'deepseek', 'gemini']
        for fallback_api in fallback_apis:
            if fallback_api != api_to_use and fallback_api in AI_APIS:
                fallback_response = api_manager.query_api(fallback_api, user_message, context)
                if fallback_response["success"]:
                    return {
                        "response": fallback_response["data"],
                        "api_used": fallback_api,
                        "knowledge_used": knowledge_used,
                        "confidence": relevant_docs[0][1] if relevant_docs else 0.0
                    }
        
        # Final fallback
        if context:
            return {
                "response": f"I found relevant information: {relevant_docs[0][0][:400]}...",
                "api_used": "knowledge_base",
                "knowledge_used": knowledge_used,
                "confidence": relevant_docs[0][1]
            }
        else:
            return {
                "response": "I'm experiencing some technical difficulties. Please try again later.",
                "api_used": "error",
                "knowledge_used": [],
                "confidence": 0.0
            }

# Enhanced Flask routes
@app.route('/chat', methods=['POST'])
def chat():
    """Enhanced chat endpoint with multiple AI APIs and RAG"""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({"error": "Missing required field: 'message'"}), 400
        
        message = data['message']
        api_preference = data.get('api_preference')  # Allow user to specify API
        
        if not isinstance(message, str) or not message.strip():
            return jsonify({"error": "Message must be a non-empty string"}), 400
        
        logger.info(f"Processing chat message: {message[:100]}...")
        
        # Generate enhanced response
        result = generate_enhanced_response(message.strip(), api_preference)
        
        # Log conversation
        conversation_id = str(uuid.uuid4())
        conn = sqlite3.connect(kb_manager.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO conversations (id, user_message, bot_response, api_used, knowledge_used, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (conversation_id, message, result['response'], result['api_used'], 
              json.dumps(result['knowledge_used']), datetime.now()))
        conn.commit()
        conn.close()
        
        return jsonify({
            "reply": result['response'],
            "api_used": result['api_used'],
            "knowledge_sources": result['knowledge_used'],
            "confidence": result['confidence'],
            "conversation_id": conversation_id
        }), 200
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/add_knowledge', methods=['POST'])
def add_knowledge():
    """Add new knowledge to the knowledge base"""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        
        required_fields = ['title', 'content']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        doc_id = kb_manager.add_document(
            title=data['title'],
            content=data['content'],
            category=data.get('category', 'user_added'),
            source=data.get('source', 'api'),
            metadata=data.get('metadata', {})
        )
        
        return jsonify({
            "success": True,
            "document_id": doc_id,
            "message": "Knowledge added successfully"
        }), 200
        
    except Exception as e:
        logger.error(f"Error in add_knowledge endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/add_training_data', methods=['POST'])
def add_training_data():
    """Add training data for model improvement"""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        
        required_fields = ['input_text', 'expected_output']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        training_id = kb_manager.add_training_data(
            input_text=data['input_text'],
            expected_output=data['expected_output'],
            category=data.get('category', 'general')
        )
        
        return jsonify({
            "success": True,
            "training_id": training_id,
            "message": "Training data added successfully"
        }), 200
        
    except Exception as e:
        logger.error(f"Error in add_training_data endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/upload_knowledge', methods=['POST'])
def upload_knowledge():
    """Upload knowledge from files (txt, json, csv)"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process file based on extension
        file_ext = filename.lower().split('.')[-1]
        
        if file_ext == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            doc_id = kb_manager.add_document(
                title=filename,
                content=content,
                category='uploaded',
                source='file_upload'
            )
            
        elif file_ext == 'json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                doc_ids = []
                for item in data:
                    if 'title' in item and 'content' in item:
                        doc_id = kb_manager.add_document(
                            title=item['title'],
                            content=item['content'],
                            category=item.get('category', 'uploaded'),
                            source='file_upload'
                        )
                        doc_ids.append(doc_id)
                doc_id = doc_ids
            else:
                doc_id = kb_manager.add_document(
                    title=data.get('title', filename),
                    content=str(data),
                    category='uploaded',
                    source='file_upload'
                )
        
        elif file_ext == 'csv':
            df = pd.read_csv(file_path)
            doc_ids = []
            
            for _, row in df.iterrows():
                # Try to find title and content columns
                title_col = next((col for col in df.columns if 'title' in col.lower()), df.columns[0])
                content_col = next((col for col in df.columns if 'content' in col.lower() or 'text' in col.lower()), df.columns[-1])
                
                doc_id = kb_manager.add_document(
                    title=str(row[title_col])[:100],
                    content=str(row[content_col]),
                    category='uploaded',
                    source='file_upload'
                )
                doc_ids.append(doc_id)
            
            doc_id = doc_ids
        
        else:
            return jsonify({"error": f"Unsupported file type: {file_ext}"}), 400
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return jsonify({
            "success": True,
            "document_ids": doc_id if isinstance(doc_id, list) else [doc_id],
            "message": f"Knowledge uploaded successfully from {filename}"
        }), 200
        
    except Exception as e:
        logger.error(f"Error in upload_knowledge endpoint: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/knowledge_stats', methods=['GET'])
def knowledge_stats():
    """Get statistics about the knowledge base"""
    try:
        conn = sqlite3.connect(kb_manager.db_path)
        cursor = conn.cursor()
        
        # Get document stats
        cursor.execute('SELECT COUNT(*), category FROM documents GROUP BY category')
        doc_stats = dict(cursor.fetchall())
        
        cursor.execute('SELECT COUNT(*) FROM documents')
        total_docs = cursor.fetchone()[0]
        
        # Get conversation stats
        cursor.execute('SELECT COUNT(*) FROM conversations')
        total_conversations = cursor.fetchone()[0]
        
        # Get training data stats
        cursor.execute('SELECT COUNT(*) FROM training_data')
        total_training = cursor.fetchone()[0]
        
        # Get recent activity
        cursor.execute('''
            SELECT api_used, COUNT(*) 
            FROM conversations 
            WHERE timestamp > datetime('now', '-24 hours')
            GROUP BY api_used
        ''')
        recent_api_usage = dict(cursor.fetchall())
        
        conn.close()
        
        return jsonify({
            "knowledge_base": {
                "total_documents": total_docs,
                "documents_by_category": doc_stats,
                "total_training_examples": total_training
            },
            "usage": {
                "total_conversations": total_conversations,
                "api_usage_24h": recent_api_usage
            },
            "available_apis": list(AI_APIS.keys()),
            "embedding_dimensions": kb_manager.document_embeddings.shape[1] if kb_manager.document_embeddings is not None else 0
        }), 200
        
    except Exception as e:
        logger.error(f"Error in knowledge_stats endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/search_knowledge', methods=['POST'])
def search_knowledge():
    """Search the knowledge base directly"""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        
        if 'query' not in data:
            return jsonify({"error": "Missing required field: 'query'"}), 400
        
        query = data['query']
        top_k = data.get('top_k', 5)
        threshold = data.get('threshold', 0.1)
        
        relevant_docs = kb_manager.find_relevant_documents(query, top_k, threshold)
        
        results = []
        for content, similarity, metadata in relevant_docs:
            results.append({
                "title": metadata['title'],
                "content": content[:300] + "..." if len(content) > 300 else content,
                "similarity_score": float(similarity),
                "category": metadata['category'],
                "source": metadata['source']
            })
        
        return jsonify({
            "query": query,
            "results": results,
            "total_found": len(results)
        }), 200
        
    except Exception as e:
        logger.error(f"Error in search_knowledge endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/export_knowledge', methods=['GET'])
def export_knowledge():
    """Export knowledge base as JSON"""
    try:
        conn = sqlite3.connect(kb_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT title, content, category, source, metadata FROM documents')
        documents = cursor.fetchall()
        
        cursor.execute('SELECT input_text, expected_output, category FROM training_data')
        training_data = cursor.fetchall()
        
        conn.close()
        
        export_data = {
            "documents": [
                {
                    "title": doc[0],
                    "content": doc[1],
                    "category": doc[2],
                    "source": doc[3],
                    "metadata": json.loads(doc[4] or '{}')
                }
                for doc in documents
            ],
            "training_data": [
                {
                    "input_text": train[0],
                    "expected_output": train[1],
                    "category": train[2]
                }
                for train in training_data
            ],
            "export_timestamp": datetime.now().isoformat(),
            "total_documents": len(documents),
            "total_training_examples": len(training_data)
        }
        
        return jsonify(export_data), 200
        
    except Exception as e:
        logger.error(f"Error in export_knowledge endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Provide feedback on bot responses for improvement"""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        
        required_fields = ['conversation_id', 'score']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        conversation_id = data['conversation_id']
        score = data['score']  # 1-5 rating
        comments = data.get('comments', '')
        
        # Update conversation with feedback
        conn = sqlite3.connect(kb_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE conversations 
            SET feedback_score = ? 
            WHERE id = ?
        ''', (score, conversation_id))
        
        conn.commit()
        conn.close()
        
        # If poor rating and comments provided, add as training data
        if score <= 2 and comments:
            cursor.execute('SELECT user_message FROM conversations WHERE id = ?', (conversation_id,))
            result = cursor.fetchone()
            if result:
                kb_manager.add_training_data(
                    input_text=result[0],
                    expected_output=comments,
                    category='feedback_improvement'
                )
        
        logger.info(f"Received feedback for conversation {conversation_id}: score={score}")
        
        return jsonify({
            "success": True,
            "message": "Feedback recorded successfully"
        }), 200
        
    except Exception as e:
        logger.error(f"Error in feedback endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    """Retrain/rebuild the knowledge base embeddings"""
    try:
        logger.info("Rebuilding knowledge base embeddings...")
        kb_manager.load_knowledge_base()
        
        return jsonify({
            "success": True,
            "message": "Knowledge base retrained successfully",
            "total_documents": len(kb_manager.documents),
            "embedding_dimensions": kb_manager.document_embeddings.shape[1] if kb_manager.document_embeddings is not None else 0
        }), 200
        
    except Exception as e:
        logger.error(f"Error in retrain endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api_status', methods=['GET'])
def api_status():
    """Check status of all configured APIs"""
    try:
        status_results = {}
        
        for api_name, config in AI_APIS.items():
            status = {
                "configured": bool(config.get('local', False) or config['key']),
                "free_tier": config.get('free', False),
                "local": config.get('local', False),
                "rate_limit_ok": api_manager._check_rate_limit(api_name)
            }
            
            # Test basic connectivity for local APIs
            if config.get('local', False) and status["configured"]:
                try:
                    test_response = requests.get("http://localhost:11434", timeout=2)
                    status["available"] = test_response.status_code in [200, 404]  # 404 is OK for root endpoint
                except:
                    status["available"] = False
            else:
                status["available"] = status["configured"]
            
            status_results[api_name] = status
        
        return jsonify({
            "apis": status_results,
            "recommended_api": api_manager.get_best_available_api(),
            "total_configured": sum(1 for status in status_results.values() if status["configured"])
        }), 200
        
    except Exception as e:
        logger.error(f"Error in api_status endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/conversation_history', methods=['GET'])
def conversation_history():
    """Get conversation history with pagination"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        
        offset = (page - 1) * per_page
        
        conn = sqlite3.connect(kb_manager.db_path)
        cursor = conn.cursor()
        
        # Get total count
        cursor.execute('SELECT COUNT(*) FROM conversations')
        total_conversations = cursor.fetchone()[0]
        
        # Get conversations with pagination
        cursor.execute('''
            SELECT user_message, bot_response, api_used, knowledge_used, timestamp, feedback_score
            FROM conversations
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        ''', (per_page, offset))
        
        conversations = cursor.fetchall()
        conn.close()
        
        results = []
        for conv in conversations:
            results.append({
                "user_message": conv[0],
                "bot_response": conv[1],
                "api_used": conv[2],
                "knowledge_sources": json.loads(conv[3] or '[]'),
                "timestamp": conv[4],
                "feedback_score": conv[5]
            })
        
        return jsonify({
            "conversations": results,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total_conversations,
                "total_pages": (total_conversations + per_page - 1) // per_page
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error in conversation_history endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# Serve the main chat interface
@app.route('/')
def index():
    """Serve the enhanced chatbot frontend"""
    return send_from_directory('.', 'index.html')

@app.route('/admin')
def admin():
    """Serve admin interface for knowledge management"""
    return send_from_directory('.', 'admin.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint"""
    try:
        # Test database connection
        conn = sqlite3.connect(kb_manager.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM documents')
        doc_count = cursor.fetchone()[0]
        conn.close()
        
        # Check available APIs
        available_apis = []
        for api_name, config in AI_APIS.items():
            if config.get('local', False) or config['key']:
                available_apis.append(api_name)
        
        return jsonify({
            "status": "healthy",
            "message": "Enhanced Flask chat API is running",
            "knowledge_base": {
                "documents": doc_count,
                "embeddings_ready": kb_manager.document_embeddings is not None
            },
            "apis": {
                "available": available_apis,
                "total_configured": len(available_apis)
            },
            "features": [
                "RAG with custom knowledge base",
                "Multiple AI API support",
                "Training data collection",
                "File upload support",
                "Conversation history",
                "Feedback system"
            ]
        }), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Initialize logging
    logger.info("=" * 60)
    logger.info("🚀 Starting Enhanced AI Chatbot Server")
    logger.info("=" * 60)
    
    # Log configuration
    logger.info("🔧 Configuration:")
    logger.info(f"   📚 Knowledge Base: {len(kb_manager.documents)} documents loaded")
    logger.info(f"   🤖 Available APIs: {list(AI_APIS.keys())}")
    
    configured_apis = []
    free_apis = []
    for api_name, config in AI_APIS.items():
        if config.get('local', False) or config['key']:
            configured_apis.append(api_name)
        if config.get('free', False):
            free_apis.append(api_name)
    
    logger.info(f"   ✅ Configured APIs: {configured_apis}")
    logger.info(f"   🆓 Free APIs: {free_apis}")
    
    # Log available endpoints
    logger.info("🌐 Available endpoints:")
    endpoints = [
        "POST /chat - Enhanced chat with RAG",
        "POST /add_knowledge - Add knowledge documents",
        "POST /add_training_data - Add training examples",
        "POST /upload_knowledge - Upload knowledge files",
        "GET /knowledge_stats - Knowledge base statistics",
        "POST /search_knowledge - Search knowledge base",
        "GET /export_knowledge - Export knowledge as JSON",
        "POST /feedback - Provide response feedback",
        "POST /retrain - Rebuild embeddings",
        "GET /api_status - Check API availability",
        "GET /conversation_history - View chat history",
        "GET /health - Health check",
        "GET / - Main chat interface",
        "GET /admin - Admin interface"
    ]
    for endpoint in endpoints:
        logger.info(f"   📡 {endpoint}")
    
    logger.info("=" * 60)
    logger.info("🎯 Features enabled:")
    features = [
        "🧠 Retrieval-Augmented Generation (RAG)",
        "🔀 Multiple AI API support with fallbacks",
        "📚 Custom knowledge base training",
        "📁 File upload support (TXT, JSON, CSV)",
        "💾 Persistent storage with SQLite",
        "📊 Advanced analytics and statistics",
        "🔄 Feedback loop for improvement",
        "⚡ Rate limiting and error handling",
        "🎨 Professional web interface"
    ]
    for feature in features:
        logger.info(f"   {feature}")
    
    logger.info("=" * 60)
    logger.info("💡 Setup Instructions:")
    logger.info("   1. Set API keys as environment variables:")
    logger.info("      export GEMINI_API_KEY='your_key_here'")
    logger.info("      export DEEPSEEK_API_KEY='your_key_here'")
    logger.info("      export OPENROUTER_API_KEY='your_key_here'")
    logger.info("   2. For local Ollama: install and run 'ollama serve'")
    logger.info("   3. Access chat interface at http://localhost:5000")
    logger.info("   4. Access admin panel at http://localhost:5000/admin")
    logger.info("=" * 60)
    
    # Run the enhanced Flask server
    app.run(host='0.0.0.0', port=5000, debug=True)