import torch.nn.functional as F
from tqdm import tqdm
import math
import wandb

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    # List to store batch losses
    batch_losses = []
    n_steps_per_epoch = math.ceil(len(train_loader.dataset) / args.batch_size)
    
    # Wrap train_loader with tqdm for the progress bar
    train_loader = tqdm(train_loader, desc=f'Training Epoch {epoch}', dynamic_ncols=True)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        preds = model(data)
        loss = F.nll_loss(preds, target)

        loss.backward()
        optimizer.step()

        # Append the batch loss to the list
        batch_losses.append(loss.item())

        if batch_idx % args.log_interval == 0:
            metrics = {"train_loss": loss,
                        "epoch": (batch_idx + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch}
        
        if batch_idx + 1 < n_steps_per_epoch:
            # 🐝 Log train metrics to wandb
            wandb.log(metrics)
        
    # Calculate and print the average training loss at the end of the epoch
    avg_train_loss = sum(batch_losses) / len(batch_losses)
    print(f'Epoch {epoch}, Average Training Loss: {avg_train_loss:.4f}')

