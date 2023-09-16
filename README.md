# lib-mapper-tf-model
Design and training the tflite model 

## Architecture
there will be two models
- segmentation: which shows each book-blob/cluster seperately in the image
- book recognition: which will finally map the book-spine image to its dataset

we will use siamese net to generate embbedings

## tech-concerns
- if we use siamese embedding and keep including new books, the embeddings for all books may keep changing and huge database update may be requitred each time new books are included