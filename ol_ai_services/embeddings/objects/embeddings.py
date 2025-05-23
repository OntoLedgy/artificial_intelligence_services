import numpy as np

# TODO: MKh - should we import faiss? - added to requirements faiss-cpu - DONE
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
        
        if len(documents) ==0 :
            raise ValueError("Cannot initialise embeddings with empty documents")
        
        self.documents = documents
        
        self.index_file_full_path = index_file_full_path
        
        self.file_metadata = file_metadata

    def create(self):
        article_embeddings = self.model.encode(
                self.documents,
                convert_to_tensor=False)

        # Convert embeddings to a numpy array
        article_embeddings = np.array(
                article_embeddings)


        embedding_dimension = article_embeddings.shape[1]  # Dimension of embeddings

        self.index = faiss.IndexFlatL2(
                embedding_dimension)
        
        self.index.add(
                article_embeddings)

    def save(self):
        # Save the index and article metadata for later use
        faiss.write_index(
                self.index,
                self.index_file_full_path)

        np.save(
                self.file_metadata,
                self.documents)
