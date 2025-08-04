# test_clustering.py

from clustering_utils import cluster_intents

samples = [
    "Hello world",
    "Bonjour tout le monde",
    "你好，世界",
    "How are you?"
]

print(cluster_intents(samples))