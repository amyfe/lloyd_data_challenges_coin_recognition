import torch

print("Torch version:")
print(torch.__version__)

from coin_clip import CoinClip

# Automatically download the model from Huggingface
model = CoinClip(model_name='breezedeus/coin-clip-vit-base-patch32')

print("hi")
images = ['test_coin_one_euro.png', 'test_coin_two_euro.png', 'test_coin_two_euro.png']
print("hello")
img_feats, success_ids = model.get_image_features(images)
print(img_feats.shape)