import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals

# Step 1: Generate an Erdős-Rényi graph with 50 vertices and edge probability p=0.1
n_vertices = 50
p = 0.1
G = nx.erdos_renyi_graph(n_vertices, p)

# Step 2: Compute the Laplacian matrix of the graph
laplacian_matrix = nx.laplacian_matrix(G).todense()

# Step 3: Calculate the graph energy by summing the absolute values of the Laplacian eigenvalues
eigenvalues = eigvals(laplacian_matrix)
graph_energy = np.sum(np.abs(eigenvalues))

# Step 4: Plot the graph and show energy
plt.figure(figsize=(12, 6))

# Subplot 1: Plot the Erdős-Rényi graph
plt.subplot(1, 2, 1)
nx.draw(G, node_size=50, node_color="skyblue", edge_color="gray", with_labels=False)
plt.title("Erdős-Rényi Graph (n=50, p=0.1)")

# Subplot 2: Display the topological energy
plt.subplot(1, 2, 2)
plt.bar(["Graph Energy"], [graph_energy], color="salmon")
plt.ylabel("Energy")
plt.title("Topological Energy of the Graph")

plt.tight_layout()
plt.show()
