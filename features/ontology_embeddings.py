import networkx as nx
import numpy as np
from node2vec import Node2Vec
import logging

logger = logging.getLogger(__name__)

# --- Placeholder for Ontology Data ---
# In a real scenario, this data would come from an actual ICD/ATC ontology file.
DUMMY_ICD_HIERARCHY = {
    'A00-B99': ['ROOT'], # Certain infectious and parasitic diseases
    'A00-A09': ['A00-B99'], # Intestinal infectious diseases
    'A01': ['A00-A09'],     # Typhoid and paratyphoid fevers
    'A01.0': ['A01'],       # Typhoid fever
    'C00-D49': ['ROOT'], # Neoplasms
    'C00-C14': ['C00-D49'], # Malignant neoplasms of lip, oral cavity and pharynx
    'C01': ['C00-C14'],     # Malignant neoplasm of base of tongue
    'Z00-Z99': ['ROOT'], # Factors influencing health status and contact with health services
    'Z00': ['Z00-Z99'],     # Encounter for general examination without complaint
}

def load_ontology_graph(ontology_data=None, representation_type='icd'):
    """
    Loads or builds an ontology graph (e.g., for ICD or ATC codes).

    Args:
        ontology_data (any, optional): Data source for the ontology.
                                       If None, uses a dummy ICD hierarchy.
        representation_type (str): Type of ontology ('icd', 'atc'). For future use.

    Returns:
        nx.DiGraph: A NetworkX directed graph representing the ontology.
    """
    graph = nx.DiGraph()
    data_to_process = ontology_data if ontology_data is not None else DUMMY_ICD_HIERARCHY

    if isinstance(data_to_process, dict):
        for child, parents in data_to_process.items():
            if not isinstance(parents, list):
                parents = [parents]
            for parent in parents:
                graph.add_edge(str(parent), str(child))
    else:
        logger.warning("Ontology data format not recognized or data is empty. Using empty graph.")

    logger.info(f"Ontology graph created with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    return graph

def train_node2vec_embeddings(graph, dimensions=64, walk_length=30, num_walks=200, workers=4, p=1, q=1, window=10, min_count=1, batch_words=4):
    """
    Trains Node2Vec embeddings on the ontology graph.

    Args:
        graph (nx.Graph or nx.DiGraph): The ontology graph.
        All other parameters are for the Node2Vec algorithm.

    Returns:
        dict: A dictionary mapping node IDs (codes) to their embedding vectors (np.ndarray).
    """
    if not graph.nodes():
        logger.warning("Graph has no nodes. Cannot train Node2Vec.")
        return {}

    logger.info(f"Training Node2Vec with dimensions={dimensions} on a graph with {graph.number_of_nodes()} nodes.")

    # Node2Vec requires string node labels
    graph_str_nodes = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})

    node2vec_model = Node2Vec(graph_str_nodes, dimensions=dimensions, walk_length=walk_length,
                              num_walks=num_walks, p=p, q=q, workers=workers, quiet=True)

    wv_model = node2vec_model.fit(window=window, min_count=min_count, batch_words=batch_words)

    embeddings = {node_id: wv_model.wv[node_id] for node_id in wv_model.wv.index_to_key}
    logger.info(f"Node2Vec training complete. Embeddings generated for {len(embeddings)} nodes.")
    return embeddings


class OntologyEmbedder:
    def __init__(self, ontology_data=None, representation_type='icd', embedding_dim=64,
                 node2vec_params=None, pretrained_embeddings_path=None):
        """
        Manages ontology graph and code embeddings.
        """
        self.graph = load_ontology_graph(ontology_data, representation_type)
        self.embedding_dim = embedding_dim
        self.embeddings = {}

        if pretrained_embeddings_path:
            self.load_embeddings(pretrained_embeddings_path)
        elif self.graph.number_of_nodes() > 0:
            n2v_params = node2vec_params if node2vec_params else {}
            n2v_params.setdefault('dimensions', self.embedding_dim)
            self.embeddings = train_node2vec_embeddings(self.graph, **n2v_params)
        else:
            logger.warning("No ontology graph loaded and no pre-trained embeddings provided.")

        self.unknown_embedding = np.zeros(self.embedding_dim)

    def get_embedding(self, code):
        """
        Retrieves the embedding for a given code.
        """
        return self.embeddings.get(str(code), self.unknown_embedding)

    def get_embeddings_for_codes(self, codes):
        """
        Retrieves embeddings for a list of codes.
        """
        if not codes:
            return np.array([])
        return np.array([self.get_embedding(code) for code in codes])

    def save_embeddings(self, filepath):
        """Saves embeddings to a file (e.g., .npz format)."""
        np.savez_compressed(filepath, **self.embeddings)
        logger.info(f"Embeddings saved to {filepath}")

    def load_embeddings(self, filepath):
        """Loads embeddings from a file."""
        try:
            loaded_data = np.load(filepath)
            self.embeddings = {key: loaded_data[key] for key in loaded_data.files}
            if self.embeddings:
                first_key = list(self.embeddings.keys())[0]
                self.embedding_dim = self.embeddings[first_key].shape[0]
                self.unknown_embedding = np.zeros(self.embedding_dim)
            logger.info(f"Embeddings loaded from {filepath} for {len(self.embeddings)} codes.")
        except FileNotFoundError:
            logger.error(f"Error: Pretrained embeddings file not found at {filepath}")
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("--- Ontology Embedding Example ---")

    # 1. Initialize OntologyEmbedder
    logger.info("\n1. Initializing OntologyEmbedder with dummy ICD data...")
    embedder = OntologyEmbedder(embedding_dim=64)

    # 2. Get embedding for a specific code
    logger.info("\n2. Getting embedding for a known code ('A01')...")
    code_a01 = 'A01'
    embedding_a01 = embedder.get_embedding(code_a01)
    logger.info(f"Embedding for '{code_a01}' (shape {embedding_a01.shape}):\n {embedding_a01[:5]}...")

    # 3. Get embeddings for a list of codes
    logger.info("\n4. Getting embeddings for a list of codes...")
    codes_list = ['A01.0', 'C01', 'UNKNOWN_CODE', 'Z00']
    embeddings_list = embedder.get_embeddings_for_codes(codes_list)
    logger.info(f"Embeddings matrix shape for list: {embeddings_list.shape}")
    for code, emb in zip(codes_list, embeddings_list):
        logger.info(f" - Code: {code}, Embedding (first 5 dims): {emb[:5]}")

    # 5. Save and Load Embeddings
    embeddings_filepath = "dummy_ontology_embeddings.npz"
    logger.info(f"\n5. Saving embeddings to {embeddings_filepath}...")
    embedder.save_embeddings(embeddings_filepath)

    logger.info(f"\n6. Loading embeddings from {embeddings_filepath} into a new embedder...")
    new_embedder = OntologyEmbedder(embedding_dim=64, pretrained_embeddings_path=embeddings_filepath)
    loaded_embedding_a01 = new_embedder.get_embedding(code_a01)
    if np.allclose(embedding_a01, loaded_embedding_a01):
        logger.info(f"Successfully loaded and verified embedding for '{code_a01}'.")
    else:
        logger.error(f"Error: Mismatch in loaded embedding for '{code_a01}'.")

    import os
    if os.path.exists(embeddings_filepath):
        os.remove(embeddings_filepath)
        logger.info(f"Cleaned up {embeddings_filepath}.")

    logger.info("\n--- Example Finished ---")
