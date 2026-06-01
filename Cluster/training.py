import argparse

import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from Cluster.utils.dataHandling import DataProvider
from Cluster.utils.lossFunctions import LossFns

def train(args: argparse.Namespace, dataloader, model: torch.nn.Module, loss_fn: LossFns,
          optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler,
          scaler: torch.amp.GradScaler):
    
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)

    model.train()
    train_loss = 0

    for X, _ in dataloader:

        X = X.to(device)
        optimizer.zero_grad()

        # Runs the forward pass in mixed precision
        with torch.amp.autocast(device_type=device_type):
            loss = loss_fn.loss(model=model, mini_batch=X)

        # Scales the loss and completes the backward pass
        scaler.scale(loss).backward()

        # Unscale gradients before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Step optimizer and scaler
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Detach removes the tensor from the computation graph to save memory, 
        # but keeps the value on the GPU, avoiding CPU synchronization.
        train_loss += loss.detach()

        if args.proof_of_concept:
            break

    print(f"\n Train Avg loss: {train_loss.item() / len(dataloader):>8f} \n")

def test(args: argparse.Namespace, dataloader, model: torch.nn.Module, loss_fn: object):    # type: ignore    due to type of dataloader partially unknown warning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_batches = len(dataloader)

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)

            test_loss += loss_fn.loss(model=model, mini_batch=X).detach()

            if args.proof_of_concept:
                break
        
    print(f"Test Avg loss: {test_loss.item() / num_batches:>8f} \n")


def training_wrapper(args: argparse.Namespace, loss_fn: object, model: torch.nn.Module, data: DataProvider, save_path: str):

    train_dataloader, test_dataloader = data.get_datasets_for_training()

    print('\nSetting optimizer and learning rates')
    # Define step counts
    epochs = args.epochs
    batches_per_epoch = len(train_dataloader)    # type: ignore
    total_steps = epochs * batches_per_epoch

    # Dedicate the first 5% of total iterations to warming up
    warmup_steps = int(total_steps * 0.05) 

    # 1. Initialize optimizer with the TARGET scaled learning rate
    target_lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=target_lr)

    # 2. Warmup: Linearly increase LR from near-zero (target_lr * 1e-8) to target_lr
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps
    )

    # 3. Decay: Cosine annealing for the remainder of training
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(total_steps - warmup_steps),
        eta_min=1e-5
    )

    # 4. Chain the schedulers
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_steps]
    )
    
    # Initialize the Gradient Scaler for AMP
    print('\nInitialize the amp grad scaler')
    scaler = torch.amp.GradScaler(device='cuda' if torch.cuda.is_available() else 'cpu')

    # train the model
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")

        train(args, train_dataloader, model, loss_fn, optimizer, scheduler, scaler)
        test(args, test_dataloader, model, loss_fn)

        print(f"LR after epoch {epoch+1}: {scheduler.get_last_lr()[0]:.6f}")

    print("Done!")

    uncompiled_model = getattr(model, "_orig_mod", model)
    torch.save(uncompiled_model.state_dict(), save_path)
    print('Saving the model')