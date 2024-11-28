# Import necessary libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns

# Part 1: Data Preparation and Visualization
print("Part 1: Data Preparation and Exploration\n")

def load_filepaths(target_dir):
    """
    Load file paths from the given directory and its subdirectories.
    Arguments:
        target_dir: Directory path containing the image data
    Returns:
        List of file paths
    """
    paths = []
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(root, file))
    return paths

def prepare_data(target_dir):
    """
    Prepare data by collecting file paths and labels.
    Assumes each subdirectory in target_dir corresponds to a class.
    Arguments:
        target_dir: Root directory containing class subdirectories
    Returns:
        filepaths: Array of file paths
        labels: Tensor of corresponding labels
        class_names: List of class names
    """
    filepaths = []
    labels = []
    class_names = os.listdir(target_dir)
    # Filter out any files, only keep directories
    class_names = [d for d in class_names if os.path.isdir(os.path.join(target_dir, d))]
    class_names.sort()  # Ensure consistent class ordering
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    for cls_name in class_names:
        cls_dir = os.path.join(target_dir, cls_name)
        fpaths = load_filepaths(cls_dir)
        labels += [class_to_idx[cls_name]] * len(fpaths)
        filepaths += fpaths

    return np.array(filepaths), torch.tensor(labels), class_names

# Load training and testing data
print("Loading and preparing dataset...")
train_dir = 'train'
test_dir = 'test'

train_filepaths, train_labels, class_names = prepare_data(train_dir)
test_filepaths, test_labels, _ = prepare_data(test_dir)

print(f"Classes: {class_names}")
print(f"Number of training samples: {len(train_filepaths)}")
print(f"Number of testing samples: {len(test_filepaths)}\n")

# Enhanced data loading with augmentation
def load_images(filepaths, train=True):
    """
    Load and preprocess images with augmentation for training.
    Arguments:
        filepaths: List of image file paths
        train: Boolean indicating if this is for training (enables augmentation)
    Returns:
        Tensor of preprocessed images
    """
    if train:
        # Training transform with data augmentation
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        # Testing transform without augmentation
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    tensors = []
    for item in filepaths:
        image = Image.open(item).convert('RGB')
        img_tensor = transform(image)
        tensors.append(img_tensor.unsqueeze(0))

    return torch.cat(tensors, dim=0)

def visualize_data_augmentation(filepath):
    """
    Visualize the effects of data augmentation on a single image.
    Arguments:
        filepath: Path to the image file
    """
    # Load original image
    image = Image.open(filepath).convert('RGB')

    # Create figure with subplots - changed to 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Display original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Define augmentations to visualize
    transforms_list = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2),
        transforms.ColorJitter(contrast=0.2),
        transforms.ColorJitter(saturation=0.2)
    ]
    titles = ['Horizontal Flip', 'Rotation', 'Brightness', 'Contrast', 'Saturation']

    # Apply and display each augmentation
    for idx, (transform, title) in enumerate(zip(transforms_list, titles)):
        # Calculate subplot position
        row = (idx + 1) // 3
        col = (idx + 1) % 3

        # Apply transformation and display
        augmented = transform(image)
        axes[row, col].imshow(augmented)
        axes[row, col].set_title(title)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

# Visualize data augmentation
print("Visualizing data augmentation effects...")
sample_filepath = train_filepaths[0]
visualize_data_augmentation(sample_filepath)

# Part 2: Model Architecture
print("\nPart 2: Model Architecture\n")

class ImprovedCNN(nn.Module):
    """
    Enhanced CNN architecture with modern improvements:
    - Deeper architecture with 3 convolutional blocks
    - Batch normalization for training stability
    - Dropout for regularization
    - Dense connections in classifier
    """
    def __init__(self, num_classes=4):
        super(ImprovedCNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )

        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )

        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Apply convolutional blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Apply classifier
        x = self.classifier(x)
        return x

# Part 3: Training and Validation Functions
print("Part 3: Training and Validation Functions\n")

def train(model, criterion, optimizer, train_filepaths, train_labels,
          val_filepaths, val_labels, device, batch_size=32, n_epochs=10):
    """
    Train the model with validation monitoring and early stopping.

    Arguments:
        model: The CNN model to train
        criterion: Loss function
        optimizer: Optimization algorithm
        train_filepaths, train_labels: Training data
        val_filepaths, val_labels: Validation data
        device: Device to run on (CPU/GPU)
        batch_size: Number of samples per batch
        n_epochs: Number of training epochs
    """
    # Initialize learning rate scheduler and early stopping parameters
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    # Lists to store metrics for plotting
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = len(train_filepaths)

        # Training loop
        for i in range(0, total_samples, batch_size):
            batch_filepaths = train_filepaths[i:i + batch_size]
            batch_labels = train_labels[i:i + batch_size]

            # Load and prepare batch
            inputs = load_images(batch_filepaths, train=True).to(device)
            labels = batch_labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track statistics
            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels).item()

            # Print progress
            if (i // batch_size) % 10 == 0:
                print(f"Epoch [{epoch+1}/{n_epochs}], Step [{i//batch_size}], Loss: {loss.item():.4f}")

        # Calculate training metrics
        epoch_loss = running_loss / total_samples
        epoch_acc = correct_preds / total_samples * 100
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Validation step
        val_loss, val_acc = validate(model, criterion, val_filepaths, val_labels, device, batch_size)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Print epoch statistics
        print(f"Epoch [{epoch+1}/{n_epochs}] - "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # Plot training history
    plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies)

    return train_losses, train_accuracies, val_losses, val_accuracies

def validate(model, criterion, val_filepaths, val_labels, device, batch_size=32):
    """
    Validate the model on validation data.

    Arguments:
        model: The CNN model to validate
        criterion: Loss function
        val_filepaths, val_labels: Validation data
        device: Device to run on (CPU/GPU)
        batch_size: Number of samples per batch

    Returns:
        val_loss: Average validation loss
        val_acc: Validation accuracy
    """
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_samples = len(val_filepaths)

    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_filepaths = val_filepaths[i:i + batch_size]
            batch_labels = val_labels[i:i + batch_size]

            inputs = load_images(batch_filepaths, train=False).to(device)
            labels = batch_labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels).item()

    val_loss = running_loss / total_samples
    val_acc = correct_preds / total_samples * 100

    return val_loss, val_acc

def plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies):
    """
    Plot training and validation metrics history.

    Arguments:
        train_losses, train_accuracies: Lists of training metrics
        val_losses, val_accuracies: Lists of validation metrics
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

def test(model, test_filepaths, test_labels, device, class_names, batch_size=32):
    """
    Evaluate the trained model on the test dataset.
    This function provides comprehensive evaluation metrics and visualizations.

    Arguments:
        model: The trained CNN model
        test_filepaths: Numpy array of test image file paths
        test_labels: Tensor of test labels
        device: Device to run on (CPU/GPU)
        class_names: List of class names
        batch_size: Number of samples per batch

    Returns:
        test_loss: Average test loss
        test_accuracy: Overall test accuracy
        conf_matrix: Confusion matrix
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()
    total_samples = len(test_filepaths)

    print("\nEvaluating model on test set...")

    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_filepaths = test_filepaths[i:i + batch_size]
            batch_labels = test_labels[i:i + batch_size]

            # Load and prepare batch
            inputs = load_images(batch_filepaths, train=False).to(device)
            labels = batch_labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Track metrics
            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)

            # Store predictions and labels for confusion matrix
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate final metrics
    test_loss = running_loss / total_samples
    test_accuracy = sum(1 for x, y in zip(all_preds, all_labels) if x == y) / len(all_labels) * 100

    # Create and plot confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(conf_matrix, class_names)

    # Print detailed metrics
    print(f"\nTest Results:")
    print(f"Overall Test Loss: {test_loss:.4f}")
    print(f"Overall Test Accuracy: {test_accuracy:.2f}%")
    print("\nPer-class metrics:")

    # Calculate and display per-class metrics
    for i, class_name in enumerate(class_names):
        precision = precision_score(all_labels, all_preds, average=None)[i]
        recall = recall_score(all_labels, all_preds, average=None)[i]
        f1 = f1_score(all_labels, all_preds, average=None)[i]

        print(f"\n{class_name}:")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-score: {f1:.3f}")

    return test_loss, test_accuracy, conf_matrix

def plot_confusion_matrix(conf_matrix, class_names):
    """
    Plot the confusion matrix as a heatmap.

    Arguments:
        conf_matrix: Confusion matrix
        class_names: List of class names
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, test_filepaths, test_labels, class_names, device, num_samples=5):
    """
    Visualize model predictions on random test samples.

    Arguments:
        model: The trained CNN model
        test_filepaths: Test image file paths
        test_labels: True labels
        class_names: List of class names
        device: Device to run on
        num_samples: Number of samples to visualize
    """
    model.eval()
    indices = np.random.choice(len(test_filepaths), num_samples, replace=False)

    plt.figure(figsize=(15, 3))
    with torch.no_grad():
        for idx, i in enumerate(indices):
            # Load and process image
            image = Image.open(test_filepaths[i])
            input_tensor = load_images([test_filepaths[i]], train=False).to(device)

            # Get prediction
            output = model(input_tensor)
            _, pred = torch.max(output, 1)

            # Plot image with labels
            plt.subplot(1, num_samples, idx + 1)
            plt.imshow(image)
            plt.axis('off')
            true_label = class_names[test_labels[i]]
            pred_label = class_names[pred.item()]
            color = 'green' if true_label == pred_label else 'red'
            plt.title(f'True: {true_label}\nPred: {pred_label}', color=color)

    plt.tight_layout()
    plt.show()

# Part 4: Main Execution
print("\nPart 4: Main Execution\n")

def main():
    """
    Main execution function that ties everything together.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model and move to device
    model = ImprovedCNN(num_classes=len(class_names)).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Split training data into train and validation sets
    train_size = int(0.8 * len(train_filepaths))
    indices = np.random.permutation(len(train_filepaths))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_paths = train_filepaths[train_indices]
    train_labs = train_labels[train_indices]
    val_paths = train_filepaths[val_indices]
    val_labs = train_labels[val_indices]

    print("Starting training...")
    # Train the model
    train_losses, train_accuracies, val_losses, val_accuracies = train(
        model, criterion, optimizer, train_paths, train_labs,
        val_paths, val_labs, device, batch_size=32, n_epochs=10
    )

    print("\nTraining completed. Loading best model for testing...")
    # Load best model for testing
    model.load_state_dict(torch.load('best_model.pth'))

    # Test the model
    test_loss, test_accuracy, conf_matrix = test(
        model, test_filepaths, test_labels, device, class_names
    )

    # Visualize some predictions
    print("\nVisualizing random test predictions...")
    visualize_predictions(model, test_filepaths, test_labels, class_names, device)

if __name__ == "__main__":
    main()
