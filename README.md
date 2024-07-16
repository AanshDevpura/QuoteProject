# Quote Finder and Generator from Image

## Link

Ollama does not work in Spaces so no Real Quotes to AI Generated Quotes.

https://huggingface.co/spaces/adevpura/QuoteProject

## Components

### Image to Text

The neural network uses encoder-decoder architecture with an attention network for the image to keywords.

Dataset is Flickr30k. In the caption, stopwords and punction were removed and common words were mapped to numbers for the neural network.

The model was trained for only 5 epochs because it took ~18 hours per epoch on my computer.

If you choose to train it more, you must run dowload_data.py first.

Beam search is used for keyword generation.

### Text to Real Quotes

Dataset: https://www.kaggle.com/datasets/akmittal/quotes-dataset

Embedding Model: BAAI/bge-large-zh-v1.5

The dataset was put through the embedding model.

This allows dot product similarity to be used to find similar quotes (like a vector database).

### Real Quotes to AI Generated Quotes

The real quotes and keywords/captions are passed to llama3 to generate a quote.

You need to download ollama for this to work.

### Other components

Users can use Salesforce/blip-image-captioning-large instead of the Neural Network for Image to Text.

Users can input their own text to bypass Image to Text.

Users can filter quotes by popularity and modify temperature for llama3.

