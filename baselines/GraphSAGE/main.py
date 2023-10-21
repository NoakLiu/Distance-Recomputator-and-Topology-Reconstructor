import torch.nn as nn
import torch
from data_preprocessing import load_data
import numpy as np
import random
from models import GraphSAGE, SupervisedGraphSage

# Define the training loop
def train(model, features, labels, idx_train, idx_val, optimizer, criterion, epochs=100):
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        optimizer.zero_grad()  # Clear previous gradients

        nodes_batch = idx_train
        outputs = model(nodes_batch)

        loss = criterion(outputs, labels[nodes_batch])
        loss.backward()  # Calculate gradients
        optimizer.step()  # Update weights

        train_acc = evaluate(model, idx_train, labels)
        val_acc = evaluate(model, idx_val, labels)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    return model

def evaluate(model, nodes, labels):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(nodes)
        predictions = outputs.argmax(1)  # Get the class with the highest probability
        correct = (predictions == labels[nodes]).sum().item()
        accuracy = correct / len(nodes)
    return accuracy

if __name__ == "__main__":
    np.random.seed(1)
    random.seed(1)

    adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset="pubmed", mode="dict")

    enc_model = GraphSAGE(num_layers=5, input_dim=features.shape[1], output_dim=128, adj_lists=adj, feat_data=features)
    model = SupervisedGraphSage(int(labels.max()) + 1, enc_model())

    # Assuming labels are already in torch.Tensor format. If not, convert them.
    labels = torch.LongTensor(labels)

    # Define the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    model = train(model, features, labels, idx_train, idx_val, optimizer, criterion)

    # Evaluate on test set
    test_acc = evaluate(model, idx_test, labels)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

