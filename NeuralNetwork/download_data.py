import deeplake
import torch
from torchvision import transforms
import os
import string
import pickle
from stop_words import get_stop_words

# Load the dataset
ds = deeplake.load('hub://activeloop/flickr30k')

# Create 'data' folder if it does not exist
os.makedirs('data', exist_ok=True)

stop_words = get_stop_words('english')

all_captions = []
for i in range(len(ds['caption_0'])):
    for j in range(5):
        # Convert numpy array to string, split into words, and convert to lowercase
        caption_words = ds[f'caption_{j}'][i].numpy()[0].split()
        lowercase_words = [word.lower() for word in caption_words]
        # Remove punctuation and stop words
        cleaned_words = [
            word.translate(str.maketrans('', '', string.punctuation))
            for word in lowercase_words
            if word not in stop_words
        ]
        # Filter out any empty strings resulting from punctuation removal
        cleaned_words = [word for word in cleaned_words if word]
        all_captions.append(cleaned_words)

# Count frequency of words
word_counts = {}
for caption in all_captions:
    for word in caption:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1

# Only keep words that appear more than 5 times
common_words = {word: count for word, count in word_counts.items() if count > 5}

# Create a word map
word_map = {word: idx for idx, word in enumerate(common_words)}
word_map['<unk>'] = len(word_map)
word_map['<start>'] = len(word_map)
word_map['<end>'] = len(word_map)
word_map['<pad>'] = len(word_map)

# Encode captions and pad them to the same length
max_length = float('-inf')
for i in range(len(all_captions)):
    max_length = max(max_length, len(all_captions[i]))

encoded_captions = []
for i in all_captions:
    encoded_captions.append([word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in i] + [word_map['<end>']] + [word_map['<pad>']] * (max_length - len(i)))

#save word_map and encodeded_captions
pickle.dump(word_map, open('word_map.pkl', 'wb'))
torch.save(torch.tensor(encoded_captions), 'data/encoded_captions.pt')


# Define the transform to resize images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Process images
images = []
for i in ds['image']:
    i = i.numpy()
    i = transform(i)
    images.append(i)

# Convert images to a tensor
images_tensor = torch.stack(images)

# Save images tensor
torch.save(images_tensor, 'data/images.pt')