import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNet:
    def __init__(self, input_size, hidden_size, output_size, inspect=False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.inspect = inspect
        
        # Initialize weights and biases randomly
        self.W1 = torch.randn(input_size, hidden_size, requires_grad=True)
        self.b1 = torch.randn(hidden_size, requires_grad=True)
        self.W2 = torch.randn(hidden_size, output_size, requires_grad=True)
        self.b2 = torch.randn(output_size, requires_grad=True)
        
        if self.inspect:
            print("Weight matrices shapes:")
            print("W1:", self.W1.shape)
            print("b1:", self.b1.shape)
            print("W2:", self.W2.shape)
            print("b2:", self.b2.shape)
        
    def forward(self, x):
        if self.inspect:
            print("Input shape:", x.shape)
        
        # Hidden layer
        hidden = torch.matmul(x, self.W1) + self.b1
        hidden = torch.sigmoid(hidden)
        
        if self.inspect:
            print("Hidden layer shape:", hidden.shape)
        
        # Output layer
        output = torch.matmul(hidden, self.W2) + self.b2
        
        if self.inspect:
            print("Output shape:", output.shape)
        
        return output

def train(model, X, y, epochs, lr, optimizer_choice, scheduler_choice):
    # Define the loss function
    criterion = nn.MSELoss()
    
    # Choose the optimizer
    if optimizer_choice == 'sgd':
        optimizer = optim.SGD([model.W1, model.b1, model.W2, model.b2], lr=lr)
    elif optimizer_choice == 'adam':
        optimizer = optim.Adam([model.W1, model.b1, model.W2, model.b2], lr=lr)
    else:
        raise ValueError("Invalid optimizer choice. Choose 'sgd' or 'adam'.")
    
    # Choose the scheduler (optional)
    if scheduler_choice == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_choice == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif scheduler_choice is None:
        scheduler = None
    else:
        raise ValueError("Invalid scheduler choice. Choose 'step', 'exponential', or None.")
    
    # Training loop
    for epoch in range(epochs):
        # Forward pass
        y_pred = model.forward(X)
        
        # Compute the loss
        loss = criterion(y_pred, y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Step the scheduler (if applicable)
        if scheduler is not None:
            scheduler.step()
        
        # Print the loss for every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Set the hyperparameters
input_size = 10
hidden_size = 20
output_size = 5
epochs = 100
lr = 0.01
optimizer_choice = 'adam'  # Choose 'sgd' or 'adam'
scheduler_choice = 'step'  # Choose 'step', 'exponential', or None
inspect = True  # Set to True to print matrix shapes, False otherwise

# Create an instance of the SimpleNet class
model = SimpleNet(input_size, hidden_size, output_size, inspect)

# Generate dummy input data and target labels
X = torch.randn(100, input_size)
y = torch.randn(100, output_size)

# Train the model
train(model, X, y, epochs, lr, optimizer_choice, scheduler_choice)