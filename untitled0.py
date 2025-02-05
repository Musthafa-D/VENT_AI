import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()
G.add_edges_from([(0, 1), (1, 2), (2, 0)])

# Define positions of nodes
pos = nx.spring_layout(G)

# Draw the graph with different arrow styles
plt.figure(figsize=(12, 8))

# Draw with default arrowstyle
plt.subplot(231)
nx.draw_networkx(G, pos, arrows=True)
plt.title("Default arrowstyle")

# Draw with '->' arrowstyle
plt.subplot(232)
nx.draw_networkx(G, pos, arrows=True, arrowstyle='->')
plt.title("Arrowstyle: '->'")

# Draw with '-[' arrowstyle
plt.subplot(233)
nx.draw_networkx(G, pos, arrows=True, arrowstyle='-[', arrowsize=20)
plt.title("Arrowstyle: '-['")

# Draw with '<-' arrowstyle
plt.subplot(234)
nx.draw_networkx(G, pos, arrows=True, arrowstyle='<-', arrowsize=20)
plt.title("Arrowstyle: '<-'")

# Draw with 'fancy' arrowstyle
plt.subplot(235)
nx.draw_networkx(G, pos, arrows=True, arrowstyle='fancy', arrowsize=20)
plt.title("Arrowstyle: 'fancy'")

plt.tight_layout()
plt.show()
