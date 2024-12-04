from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from sentrev.evaluator import evaluate_rag
import os

# Load all the embedding models
encoder3 = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device="cuda")
encoder5 = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2', device="cuda")
encoder6 = SentenceTransformer('sentence-transformers/LaBSE', device="cuda")

# Create a list of the encoders
encoders = [encoder3, encoder5, encoder6]

# Create a dictionary that maps each encoder to its name
encoder_to_names = {
    encoder3: 'all-mpnet-base-v2',
    encoder5: 'all-MiniLM-L12-v2',
    encoder6: 'LaBSE',
}

pdfs = ["data/attention_is_all_you_need.pdf", "data/generative_adversarial_nets.pdf"]

client = QdrantClient("http://localhost:6333")
distances = ["cosine", "dot", "euclid", "manhattan"]

for chunking_size in range(500,2000,500):
    for text_percentage in range(40, 100, 20):
        perc = text_percentage/100
        for distance in distances:
            os.makedirs(f"eval/{chunking_size}_{text_percentage}_{distance}/")
            csv_path = f"eval/{chunking_size}_{text_percentage}_{distance}/stats.csv"
            evaluate_rag(pdfs, encoders, encoder_to_names, client, csv_path, chunking_size, text_percentage=perc, distance=distance, mrr=10, carbon_tracking="AUT", plot=True)