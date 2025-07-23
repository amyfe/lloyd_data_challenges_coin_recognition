
import numpy as np

def get_coinclip_embeddings(image_path, coin_side='reverse', group= None):
    #Rename to find right path
    side_category_str = f"{coin_side}_{group}"
    file_path_emb = f'img_feats_coinclip/img_features_{side_category_str}.csv'

    # Load the CSV file back into a numpy array
    loaded_features = np.loadtxt(file_path_emb, delimiter=',')
    return loaded_features