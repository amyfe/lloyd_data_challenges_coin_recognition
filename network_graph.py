import plotly.graph_objects as go
import random
import networkx as nx
import numpy as np
import os
from sklearn.preprocessing import normalize

def get_embeddings(model='siamese',coin_side='reverse', group= None):
    #Rename to find right path
    
    #If not renamed, use the following. csv files without model in filename are siamese files
    if model == 'siamese':
        side_category_str = f"{coin_side}_{group}"
        file_path_emb = f'embeddings_result/embeddings_similarity_matrix_{side_category_str}.csv'
    elif model == 'triplet':
        side_category_str = f"{model}_{coin_side}_{group}"
        file_path_emb = f'embeddings_result/embeddings_triplet_matrix_{side_category_str}.csv'

    # Load the CSV file back into a numpy array
    loaded_features = np.loadtxt(file_path_emb, delimiter=',')
    return loaded_features

# Normalize weights for line thickness (scale between 0.5 and 5)
def normalize_weight(w, min_w, max_w, min_thick=0.5, max_thick=5):
    if max_w - min_w == 0:
        return (min_thick + 1) #
    return (min_thick + (w - min_w) * (max_thick - min_thick) / (max_w - min_w)) + 1

def get_layout(G, layout="spring"):
    if layout == "spring":
        return nx.spring_layout(G, seed=42)
    elif layout == "circular":
        return nx.circular_layout(G)
    elif layout == "kamada_kawai":
        return nx.kamada_kawai_layout(G)
    elif layout == "shell":
        return nx.shell_layout(G)
    elif layout == "spectral":
        return nx.spectral_layout(G)
    else:
        raise ValueError(f"Unknown layout: {layout}")

def create_network_graph(model="siamese",coin_side='reverse', group=None,threshold=0.9):

    # Get all file names in the folder
    folder_path = f"data/{coin_side}/{group}"  # Adjust the path as needed
    nodes_name = [os.path.join(f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    full_node_name = [os.path.join(folder_path,f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    print("Number of nodes:", len(nodes_name))

    # Load the similarity matrix
    sim_matrix = get_embeddings(model,coin_side, group)
    print("Similarity matrix shape:", sim_matrix.shape[0])
    print("Similarity matrix:", sim_matrix)
    print(f"Graph for {model}_{coin_side}_{group} is being created...")

    # Create graph from similarity matrix
    G = nx.Graph()
    for i, node_i in enumerate(nodes_name):
        G.add_node(i, name=node_i)
        for j, node_j in enumerate(nodes_name):
            if i < j:  # avoid duplicates
                weight = sim_matrix[i, j]
                weight = float(weight)  # Ensure weight is a float
                if weight >= threshold:  # only keep strong similarities
                    G.add_edge(i, j, weight=weight)


    weights = [G[u][v]['weight'] for u, v in G.edges()]
    min_w, max_w = min(weights), max(weights)

    # Generate node positions (spring layout)
    #pos = nx.spectral_layout(G, weight='weight')  # Use spectral layout for better visualization
    layout = "spring"  # Change this to any layout you prefer
    pos = get_layout(G, layout)  # You can change the layout here
    for node in G.nodes():
        G.nodes[node]['pos'] = pos[node]

    # Create edge traces with variable thickness
    edge_traces = []
    for (u, v) in G.edges():
        x0, y0 = G.nodes[u]['pos']
        x1, y1 = G.nodes[v]['pos']
        w = G[u][v]['weight']
        thickness = normalize_weight(w, min_w, max_w)
        
        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                line=dict(width=thickness, color='#888'),
                mode='lines',
                hoverinfo='skip',
                #hoverinfo='text',
                #text=f"Weight: {w:.2f}",
                #text=f"{nodes_name[u]} ↔ {nodes_name[v]}<br>Weight: {w:.2f}",
            )
        )
        
        # Invisible marker at the midpoint (to trigger hover)
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        edge_traces.append(
            go.Scatter(
                x=[mid_x], y=[mid_y],
                mode='markers',
                marker=dict(size=1, color='rgba(0,0,0,0)'),  # invisible marker
                hoverinfo='text',
                text=[f"{nodes_name[u]} ↔ {nodes_name[v]}<br>Weight: {w:.5f}"]
            )
        )

    # Create node scatter trace
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title=dict(
                text='Node Connections',
                side='right'
                ),
                xanchor='left',
            ),
            line_width=2))

    node_adjacencies = []
    node_text = []

    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f"<b>{nodes_name[node]}</b><br>"
    f"# of connections: {len(adjacencies[1])}<br>"
    f"<img src='{nodes_name[node]}' width='100'>"
    )

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    # Combine all traces
    fig = go.Figure(data=edge_traces + [node_trace],
                layout=go.Layout(
                    title=dict(
                        text=f"<br>Network graph of group {group} {coin_side} with {len(nodes_name)} nodes and threshold: " + str(threshold),
                        font=dict(size=16)
                    ),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=[ dict(
                        text="Python code: <a href='https://plotly.com/python/network-graphs/'> https://plotly.com/python/network-graphs/</a>",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    fig.show()
    threshold_str = str(threshold).replace('.', '')  # Replace '.' with '_' for file naming

    # Save the figure as an HTML file
    file_path = os.path.join("graphs", f'network_graph_{model}_{coin_side}_{group}.html')
    fig.write_html(file_path)

if __name__ == "__main__":
    print("This script generates a network graph with variable edge thickness based on weights.")
    print("The graph is displayed using Plotly and saved as an HTML file.")

    # Change these parameters as needed
    model= "siamese"  # Change to "triplet" if needed
    coin_side = "reverse"  # Change to "obverse" if needed

    # For one graph with model,group, coin side, threshold parameters
    create_network_graph(model,coin_side,group= "A", threshold=0.9)


    groups = ["A", "B","C","D", "E", "F", "H"]
    #groups = ["A", "C"]
    # For multiple groups, you can uncomment the following line, just delete the first three """
    """"
    for group in groups:
        print(f"Creating network graph for group {group}...")
        create_network_graph(model,"obverse", group, threshold=0.9)  # Adjust the path as needed
        create_network_graph(model,"reverse", group, threshold=0.9) # Adjust the path as needed """
