import faiss
import numpy as np
import pickle
import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional, Dict, Tuple

class FAISSVectorDB:
    def __init__(self, db_path: str = "./faiss_db", embedding_model: str = "text-embedding-3-small"):
        """
        Initialize FAISS Vector Database
        
        Args:
            db_path: Directory path to store FAISS index and metadata
            embedding_model: OpenAI embedding model name
        """
        self.db_path = db_path
        self.index_path = os.path.join(db_path, "faiss.index")
        self.metadata_path = os.path.join(db_path, "metadata.pkl")
        self.embedding_model = embedding_model
        
        # Load environment variables from .env file
        load_dotenv()
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        self.client = OpenAI(api_key=api_key)
        
        # Dimension for text-embedding-3-small is 1536
        self.dimension = 1536
        
        # Create directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Load or create FAISS index
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = {}
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding using OpenAI API
        
        Args:
            text: Text to embed
            
        Returns:
            numpy array of embedding
        """
        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        embedding = np.array(response.data[0].embedding, dtype='float32')
        return embedding
    
    def store_user_data(self, resume_data: str, linkedin_data: Optional[str] = None, user_id: str = None) -> bool:
        """
        Store user's resume and LinkedIn data as vectors in FAISS
        
        Args:
            resume_data: Long text containing resume information
            linkedin_data: Optional - Long text containing LinkedIn profile information
            user_id: Unique identifier for the user
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Handle empty or None values
            resume_data = resume_data.strip() if resume_data else ""
            linkedin_data = linkedin_data.strip() if linkedin_data else ""
            
            # Validate that at least one data source exists
            if not resume_data and not linkedin_data:
                print("Error: Both resume_data and linkedin_data are empty")
                return False
            
            # Build combined text based on available data
            combined_parts = []
            if resume_data:
                combined_parts.append(f"Resume: {resume_data}")
            if linkedin_data:
                combined_parts.append(f"LinkedIn Profile: {linkedin_data}")
            
            combined_text = "\n\n".join(combined_parts)
            
            # Generate embedding using OpenAI
            embedding = self._get_embedding(combined_text)
            embedding = np.array([embedding]).astype('float32')
            
            # Check if user_id already exists
            if user_id in self.metadata:
                print(f"Warning: user_id '{user_id}' already exists. Updating data.")
                # Remove old entry
                old_idx = self.metadata[user_id]['index']
                # Note: FAISS doesn't support deletion, so we'll just update metadata
                # and add new vector (old vector remains but won't be accessible)
            
            # Add to FAISS index
            current_idx = self.index.ntotal
            self.index.add(embedding)
            
            # Store metadata with actual data that was provided
            self.metadata[user_id] = {
                'index': current_idx,
                'resume_data': resume_data,
                'linkedin_data': linkedin_data,
                'embedding': embedding[0]
            }
            
            # Save to disk
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            data_sources = []
            if resume_data:
                data_sources.append("resume")
            if linkedin_data:
                data_sources.append("LinkedIn")
            
            print(f"Successfully stored {' and '.join(data_sources)} data for user_id: {user_id}")
            return True
            
        except Exception as e:
            print(f"Error storing data: {str(e)}")
            return False
    
    def retrieve_user_data(self, user_id: str) -> Optional[Dict[str, str]]:
        """
        Retrieve user's data from FAISS database based on user_id
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Dictionary containing resume_data and linkedin_data, or None if not found
        """
        try:
            if user_id not in self.metadata:
                print(f"user_id '{user_id}' not found in database")
                return None
            
            user_data = self.metadata[user_id]
            
            return {
                'user_id': user_id,
                'resume_data': user_data['resume_data'],
                'linkedin_data': user_data['linkedin_data']
            }
            
        except Exception as e:
            print(f"Error retrieving data: {str(e)}")
            return None
    
    def search_similar_users(self, query_text: str, top_k: int = 5) -> list:
        """
        Search for users with similar profiles based on query text
        
        Args:
            query_text: Text to search for similar profiles
            top_k: Number of top results to return
            
        Returns:
            List of tuples (user_id, distance, data)
        """
        try:
            # Generate query embedding using OpenAI
            query_embedding = self._get_embedding(query_text)
            query_embedding = np.array([query_embedding]).astype('float32')
            
            # Search in FAISS
            distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
            
            # Map indices back to user_ids
            results = []
            index_to_user = {v['index']: k for k, v in self.metadata.items()}
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx in index_to_user:
                    user_id = index_to_user[idx]
                    results.append((
                        user_id,
                        float(dist),
                        {
                            'resume_data': self.metadata[user_id]['resume_data'],
                            'linkedin_data': self.metadata[user_id]['linkedin_data']
                        }
                    ))
            
            return results
            
        except Exception as e:
            print(f"Error searching: {str(e)}")
            return []


