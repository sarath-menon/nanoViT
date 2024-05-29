import torch
from vit import ViT, ViTConfig
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# train config
class TrainConfig:
    num_epochs = 10000
    eval_iters = 200
    eval_interval= 500
    batch_size=32 
    lr=1e-3

# Create an instance of TrainConfig
train_cfg = TrainConfig()

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# Define the transformations for the CiFAR dataset
transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset  = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transforms)
val_dataset  = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transforms)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_cfg.batch_size,shuffle=True)
testloader = torch.utils.data.DataLoader(val_dataset, batch_size=train_cfg.batch_size,shuffle=False)

print("Size of training dataset:", len(train_dataset))
print("Size of validation dataset:", len(val_dataset))


# Create an instance of ViT model
vit_config = ViTConfig(device=device, img_size=32)
model = ViT(vit_config)
model = model.to(device)


@torch.no_grad()
def estimate_loss():
    """
    Estimate the average loss of the model over a specified number of batches.
    
    Args:
    - model: The model to evaluate.
    - num_batches: The number of batches to use for estimating the loss.
    - batch_size: The size of each batch.
    - block_size: The block size to use for each batch.
    
    Returns:
    - avg_loss: The average loss over the specified number of batches.
    """

    out = {}

    # Switch model to evaluation mode
    model.eval() 

    for mode in ['train', 'test']:

        # Initialize a tensor to store losses for each epoch
        losses = torch.zeros(train_cfg.eval_iters)

        for k in range(train_cfg.eval_iters):

            # Fetch a batch of data
            batch_x, batch_y = next(iter(trainloader))
            batch_x, batch_y  = batch_x.to(device), batch_y.to(device)
            
            # Call the model on the new batch of data
            _, loss = model(batch_x, batch_y)
            
            # Accumulate the loss
            losses[k] = loss.item()
        
        # store the average loss
        out[mode] = losses.mean()
        
    # Switch model to training mode
    model.train() 
    
    return out


# Move model parameters to the appropriate device (CPU or GPU)
model = model.to(device)

# Initialize the optimizer with model parameters and a learning rate of 0.001
optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)

# Define the loss function
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(train_cfg.num_epochs):

    if epoch % train_cfg.eval_interval == 0:
        estimated_loss = estimate_loss()
        print(f"Estimated Loss at epoch {epoch}: {estimated_loss}")

    # Fetch a batch of data
    batch_x, batch_y = next(iter(trainloader))
    batch_x, batch_y  = batch_x.to(device), batch_y.to(device)
    
    # Call the model on the new batch of data
    log_probs, loss = model(batch_x, batch_y)
    
    # Zero the gradients
    optimizer.zero_grad(set_to_none=True)
    
    # Backward pass: Compute gradient of the loss with respect to model parameters
    loss.backward()
    
    # Calling the step function on an Optimizer makes an update to its parameters
    optimizer.step()

# Compute and print loss
print('Loss: {loss.item()}')

# evaluate_model(max_new_tokens=500)

# Save the model weights
torch.save(model.state_dict(), 'weights/vit_weights.pth')