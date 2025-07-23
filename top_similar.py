import numpy as np
import os

def get_embeddings(model="siamese", coin_side='reverse', group= None):
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

def top_similar(name,group,model="siamese",coin_side="reverse",top_n=5):

    folder_path = f"data/{coin_side}/{group}"  # Adjust the path as needed
    nodes_name = [os.path.join(f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    full_node_name = [os.path.join(folder_path,f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
   

   # Load the similarity matrix
    sim_matrix = get_embeddings(model,coin_side, group)

    # Check if the name exists
    if name not in nodes_name:
        raise ValueError(f"{name} not found in names_entry.")

    idx = nodes_name.index(name)
    similarities = sim_matrix[idx]

    # Exclude itself by setting similarity to -inf
    similarities = similarities.copy()
    similarities[idx] = -np.inf

    # Get indices of top_n highest similarities
    top_indices = np.argsort(similarities)[-top_n:][::-1]  # Descending order

    # Collect results
    results = [(nodes_name[i], similarities[i]) for i in top_indices]
    return results


if __name__ == "__main__":
    print("This script generates a network graph with variable edge thickness based on weights.")
    print("The graph is displayed using Plotly and saved as an HTML file.")

    folder_path = f"data/reverse/B"  # Adjust the path as needed
    nodes_name = [os.path.join(f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    full_node_name = [os.path.join(folder_path,f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
   
    selected_name = nodes_name[5]  # Replace with the name you want to search for
    print(f"Selected name: {selected_name}")
    top_results = top_similar(selected_name, "siamese", "B", coin_side="reverse", top_n=5)
    print(f"Top similar items to {selected_name}:")
    for name, similarity in top_results:
        print(f"{name}: {similarity:.4f}")