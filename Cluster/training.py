import argparse
import copy

import torch
from torch.optim.lr_scheduler import LinearLR, ConstantLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from Cluster.utils.dataHandling import DataProvider
from Cluster.utils.lossFunctions import LossFns

def train(args: argparse.Namespace, dataloader,
          model: torch.nn.Module, ema_model: AveragedModel,
          loss_fn: LossFns, optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler.LRScheduler, scaler: torch.amp.GradScaler):
    
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

        # Update EMA model parameters
        ema_model.update_parameters(model)

        # Detach removes the tensor from the computation graph to save memory, 
        # but keeps the value on the GPU, avoiding CPU synchronization.
        train_loss += loss.detach()

        if args.proof_of_concept:
            break

    print(f"\n Train Avg loss: {train_loss.item() / len(dataloader):>8f} \n")


def test(args: argparse.Namespace, dataloader, model: torch.nn.Module, loss_fn: LossFns,    # type: ignore    due to type of dataloader partially unknown warning
         prefix: str = ''):

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
        
    avg_test_loss = test_loss.item() / num_batches
    print(f"{prefix}  Test Avg loss: {avg_test_loss:>8f} \n")
    
    return avg_test_loss


def training_wrapper(args: argparse.Namespace, loss_fn: LossFns, model: torch.nn.Module, data: DataProvider, save_path: str):

    train_dataloader, test_dataloader = data.get_datasets_for_training()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('\nSetting optimizer and learning rates')
    # Define step counts
    epochs = args.training_epochs
    batches_per_epoch = len(train_dataloader)    # type: ignore
    total_steps = epochs * batches_per_epoch
    warmup_steps = int(total_steps * 0.05) 

    # Initialize optimizer with the TARGET scaled learning rate
    target_lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=target_lr)

    # Warmup: Linearly increase LR from near-zero (target_lr * 1e-8) to target_lr
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps
    )

    # Constant learning rate after warmup
    constant_scheduler = ConstantLR(
        optimizer,
        factor=1.0,
        total_iters=1
    )

    # Chain the schedulers
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, constant_scheduler],
        milestones=[warmup_steps]
    )
    
    # Initialize the Gradient Scaler for AMP
    print('\nInitialize the amp grad scaler')
    scaler = torch.amp.GradScaler(device='cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the EMA Model
    ema_decay = 0.9999
    ema_avg_fn = get_ema_multi_avg_fn(ema_decay)
    ema_model = AveragedModel(model, device=device, multi_avg_fn=ema_avg_fn)

    # Setup the patience early halting
    patience = max(1, epochs // 100 * 10)
    best_test_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    best_model_type = "None"

    # train the model
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")

        train(args, train_dataloader, model, ema_model, loss_fn, optimizer, scheduler, scaler)


        # evaluate both the active model and the ema model
        active_loss = test(args, test_dataloader, model, loss_fn, 'active')
        ema_loss = test(args, test_dataloader, ema_model, loss_fn, 'ema')

        print(f"LR after epoch {epoch+1}: {scheduler.get_last_lr()[0]:.6f}")


        # Gate on the superior configuration for this epoch
        current_best_loss = min(active_loss, ema_loss)


        # Early Stopping Logic --------------------------------------------------------------------
        if current_best_loss < best_test_loss:
            best_test_loss = current_best_loss
            epochs_no_improve = 0
            
            # Extract and store the state dict of the superior model
            if active_loss < ema_loss:
                uncompiled_model = getattr(model, "_orig_mod", model)
                best_model_state = copy.deepcopy(uncompiled_model.state_dict())
                best_model_type = "Active"
            else:
                uncompiled_model = getattr(ema_model.module, "_orig_mod", ema_model.module)
                best_model_state = copy.deepcopy(uncompiled_model.state_dict())
                best_model_type = "EMA"
        else:
            epochs_no_improve += 1
            print(f"Early stopping counter: {epochs_no_improve} out of {patience}")
            
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered. No improvement for {patience} consecutive epochs.")
                break

    print("Done!")


    if best_model_state is not None:
        torch.save(best_model_state, save_path)
        print(f'Saving the best model ({best_model_type} weights)')
    else:
        uncompiled_model = getattr(model, "_orig_mod", model)
        torch.save(uncompiled_model.state_dict(), save_path)
        print('Saving the active model (fallback)')