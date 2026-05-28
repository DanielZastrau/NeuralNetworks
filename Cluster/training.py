import argparse

import torch
from torchvision import datasets    # type: ignore
from torch.utils.data import DataLoader
from torchvision.transforms import v2    # type: ignore

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


def train(dataloader, model: torch.nn.Module, loss_fn: object,
          optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    train_loss = 0

    for batch, (X, _) in enumerate(dataloader):
        X = X.to(device)

        loss = loss_fn.loss(model=model, mini_batch=X)

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        train_loss += loss.item()

        if batch % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Step {batch}/{len(dataloader)}, LR: {current_lr:.6f}, Loss: {loss.item():.6f}")

    print(f"\n Train Avg loss: {train_loss / len(dataloader):>8f} \n")

def test(dataloader, model: torch.nn.Module, loss_fn: object):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_batches = len(dataloader)

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)

            test_loss += loss_fn.loss(model=model, mini_batch=X).item()
    test_loss /= num_batches
    print(f"Test Avg loss: {test_loss:>8f} \n")


def training_wrapper(args: argparse.Namespace, loss_fn: object, model: torch.nn.Module, save_path: str):

    print('\nSetting the transform')
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),  # Scales to [0, 1]
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Shifts to [-1, 1]
    ])

    print('\nLoading train and test data')
    training_data = datasets.CIFAR10(
        root=args.data_dir if args.where == 'cluster' else "../data",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.CIFAR10(
        root=args.data_dir if args.where == 'cluster' else "../data",
        train=False,
        download=True,
        transform=transform
    )

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=4)    # type: ignore
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, num_workers=4)    # type: ignore

    print('\nSetting optimizer and learning rates')
    # Define step counts
    epochs = args.epochs
    batches_per_epoch = len(train_dataloader)    # type: ignore
    total_steps = epochs * batches_per_epoch

    # Dedicate the first 5% of total iterations to warming up
    warmup_steps = int(total_steps * 0.05) 

    # 1. Initialize optimizer with the TARGET scaled learning rate
    target_lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=target_lr, weight_decay=1e-4)

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
    
    # train the model
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")

        train(train_dataloader, model, loss_fn, optimizer, scheduler)
        test(test_dataloader, model, loss_fn)

        print(f"LR after epoch {epoch+1}: {scheduler.get_last_lr()[0]:.6f}")

        torch.save(model.state_dict(), save_path)
    print("Done!")