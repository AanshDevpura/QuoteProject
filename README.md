# QuoteProject

## Components

### Image to Text Neural Network

The neural network uses encoder-decoder arcitecture with an attention network for the image to keywords.

Dataset is Flickr30k. In the caption, stopwords and punction were removed and common words were mapped to numbers for the neural network.

The model was trained for only 5 epochs because it took ~18 hours per epoch in my computer.

If you choose to train it more, you must run dowload_data.py first.

Beam search is used for keyword generation.

### Text to Real Quotes

Dataset: https://www.kaggle.com/datasets/akmittal/quotes-dataset

Embedding Model: BAAI/bge-large-zh-v1.5

The dataset was put throught the embedding model.

This allows dot product similariy to be used to find similar quotes (like a vector database).

### Real Quotes to AI Generated Quotes

The real quotes and keywords/caption are passed to llama3 to generate a quote

### Other components

Users can use Salesforce/blip-image-captioning-large instead of the Neural Network for Image to Text.

Users can input their own text to bypass Image to Text.

# Users can filter quotes by popularity and modify temperature for llama3.

---

title: QuoteProject
emoji: 🐨
colorFrom: blue
colorTo: blue
sdk: gradio
sdk_version: 4.37.2
app_file: app.py
pinned: false

---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

> > > > > > > 2aceaec79b915b2d4667d1fd80fdb8bb91095fc8
