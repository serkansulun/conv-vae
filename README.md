# Variational Autoencoder

Modified to enable additional features:
- Includes a subsample of celebA dataset.
- Includes an extracted vector, defining the feature of having sunglasses
- interpolate.py : Interpolates two images
- extract_feature.py : Extracts mean vector for certain features
(e.g having sunglasses). To run this, you need the entire celebA dataset.
- add_feature.py : Adds/removes feature to photos. Currently only usable feature is sunglasses.


This is a simple variational autoencoder written in Pytorch and trained using the CelebA dataset.

The images are scaled down to 112x128, the VAE has a latent space with 200 dimensions and it was
trained for nearly 90 epochs. 

## Results

### Face transitions
![media/transition1.gif](media/transition1.gif)
![media/transition2.gif](media/transition2.gif)

### Mean face between two samples
![media/mean_face.png](media/mean_face.png)


