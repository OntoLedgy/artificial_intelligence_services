class CustomPGRetriever:
    def __init__(self, graph_store, llm, embed_model=None, similarity_top_k=3, path_depth=2):
        self.graph_store = graph_store
        self.llm = llm
        self.embed_model = embed_model
        self.similarity_top_k = similarity_top_k
        self.path_depth = path_depth
        
    def get_nodes(self, limit=10):
        """
        Fallback method to retrieve nodes from the graph store.
        Tries different methods available in Neo4jPGStore.
        
        Args:
            limit (int): Maximum number of nodes to retrieve.
            
        Returns:
            list: List of node objects.
        """
        try:
            # Try different methods that might exist in Neo4jPGStore
            if hasattr(self.graph_store, 'query'):
                query = f"MATCH (n) RETURN n LIMIT {limit}"
                return self.graph_store.query(query) or []
            elif hasattr(self.graph_store, 'get_all_nodes'):
                nodes = self.graph_store.get_all_nodes()
                return nodes[:limit] if nodes else []
            elif hasattr(self.graph_store, 'client') and hasattr(self.graph_store.client, 'query'):
                # Some graph stores have a client object with query method
                query = f"MATCH (n) RETURN n LIMIT {limit}"
                result = self.graph_store.client.query(query)
                return result if result else []
            else:
                # Return empty list if no suitable method found
                print("No suitable method found to retrieve nodes from graph store")
                return []
        except Exception as e:
            print(f"Error getting nodes: {str(e)}")
            return []