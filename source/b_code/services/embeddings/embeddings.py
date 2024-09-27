import numpy as np

# TODO: MKh - should we import faiss?
import faiss


# TODO: MKh - should we type the parameters? Is there a way of automating this?
class Embeddings:
    
    def __init__(
            self,
            model,
            documents,
            index_file_full_path,
            file_metadata):
        self.model = model
        self.documents = documents
        self.index_file_full_path = index_file_full_path
        self.file_metadata = file_metadata
    
    
    def create(
            self):
        article_embeddings = self.model.encode(
            self.documents,
            convert_to_tensor=False)
        
        # Convert embeddings to a numpy array
        article_embeddings = np.array(
            article_embeddings)
        
        # Initialize FAISS index
        embedding_dimension = article_embeddings.shape[1]  # Dimension of embeddings
        
        self.index = faiss.IndexFlatL2(
            embedding_dimension)
        
        # Add embeddings to the index
        self.index.add(
            article_embeddings)
    
    
    def save(
            self):
        # Save the index and article metadata for later use
        faiss.write_index(
            self.index,
            self.index_file_full_path)
        
        np.save(
            self.file_metadata,
            self.documents)
