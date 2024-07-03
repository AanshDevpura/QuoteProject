import torch
import torch.nn as nn
from models import EncoderCNN, DecoderRNN
from dataset import ImageCaptionDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
encoder_dim = 2048  # dimension of encoder
dropout = 0.5

word_map = pickle.load(open('word_map.pkl', 'rb'))
vocab_size = len(word_map)
torch.backends.cudnn.benchmark = True  # set to True because input sizes are constant

def train(encoder, decoder, criterion, decoder_optimizer, epoch, print_freq=1000):
    encoder.train()
    decoder.train()
    
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImageCaptionDataset('TRAIN', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    
    for i, (images, captions) in enumerate(train_loader):
        images = images.to(device)
        captions = captions.to(device)

        # Forward prop.
        images = encoder(images)
        scores, alphas = decoder(images, captions)
        targets = captions[:, 1:]

        # Calculate loss
        loss = criterion(scores.view(-1, vocab_size), targets.reshape(-1))

        # Back prop.
        decoder_optimizer.zero_grad()
        loss.backward()
        decoder_optimizer.step()

        # Print status
        if i % print_freq == 0:
            print(f'Epoch [{epoch}/{epochs}], Step [{i}/{len(train_loader)}], Training Loss: {loss.item()}')

def validate(encoder, decoder, criterion, epoch):
    encoder.eval()
    decoder.eval()
    
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = ImageCaptionDataset('VAL', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    
    val_loss = 0
    with torch.no_grad():
        for i, (images, captions) in enumerate(val_loader):
            images = images.to(device)
            captions = captions.to(device)

            # Forward prop.
            images = encoder(images)
            scores, alphas = decoder(images, captions)
            targets = captions[:, 1:]

            # Calculate loss
            loss = criterion(scores.view(-1, vocab_size), targets.reshape(-1))
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f'Epoch [{epoch}/{epochs}], Validation Loss: {val_loss}')

def test(encoder, decoder, criterion):
    encoder.eval()
    decoder.eval()
    
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = ImageCaptionDataset('TEST', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    
    test_loss = 0
    num_batches = len(test_loader)

    with torch.no_grad():
        for i, (images, captions) in enumerate(test_loader):
            images = images.to(device)
            captions = captions.to(device)

            # Forward prop.
            images = encoder(images)
            scores, alphas = decoder(images, captions)
            targets = captions[:, 1:]

            # Calculate loss
            loss = criterion(scores.view(-1, vocab_size), targets.reshape(-1))
            test_loss += loss.item()

    test_loss /= num_batches
    print(f'Test Loss: {test_loss}')

if __name__ == "__main__":
    # Initialize models
    # Load the encoder if 'encoder.pt' exists
    if os.path.exists('encoder.pt'):
        encoder = torch.load('encoder.pt')
    else:
        encoder = EncoderCNN().to(device)
        # Save the encoder
        torch.save(encoder, 'encoder.pt')
    
    # Load the decoder if 'decoder.pt' exists
    if os.path.exists('decoder.pt'):
        decoder = torch.load('decoder.pt')
    else:
        decoder = DecoderRNN(attention_dim, emb_dim, decoder_dim, vocab_size, encoder_dim, dropout).to(device)

    # Define the loss function and criterion
    criterion = nn.CrossEntropyLoss(ignore_index=word_map["<pad>"]).to(device)
    decoder_optimizer = torch.optim.Adam(params=decoder.parameters(), lr=0.0001)

    epochs = 5
    batch_size = 32
    workers = 0

    for epoch in range(epochs):
        train(encoder, decoder, criterion, decoder_optimizer, epoch)
        validate(encoder, decoder, criterion, epoch)

    test(encoder, decoder, criterion)

    # Save the decoder because its the only one that was trained
    torch.save(decoder, 'decoder.pt')
