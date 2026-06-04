import argparse
import copy
import os
import tempfile

import torch
from torch.optim.lr_scheduler import LinearLR, ConstantLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from Cluster.utils.dataHandling import DataProvider
from Cluster.utils.lossFunctions import LossFns
from Cluster.utils.stages import Stages
from Cluster.utils.sample_kac import TorchKacConstantSampler
from Cluster.utils.reversals import Reversal

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


def test(args: argparse.Namespace, dataloader, model: torch.nn.Module, loss_fn: LossFns):    # type: ignore    due to type of dataloader partially unknown warning

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

    return avg_test_loss


def compute_fid_checkpoint(args: argparse.Namespace, model: torch.nn.Module, data: DataProvider,
                           reversal_fns: Reversal, sampler: TorchKacConstantSampler, temp_dir: str) -> float:
    from Cluster.sampling import sample
    from Cluster.eval import evaluate_fid
    from torchvision.utils import save_image

    original_num_samples = args.sampling_num_samples
    args.sampling_num_samples = 5000

    try:
        samples = sample(args=args, model=model, data=data, reversal_fns=reversal_fns, sampler=sampler)
        samples = (samples + 1.0) / 2.0
        samples = samples.clamp(0.0, 1.0)

        samples_path = os.path.join(temp_dir, "checkpoint_samples")
        os.makedirs(samples_path, exist_ok=True)

        for i, img in enumerate(samples):
            img_path = os.path.join(samples_path, f"sample_{i:05d}.png")
            save_image(img, img_path)

        fid_score = evaluate_fid(args=args, data=data, path_to_generated_samples=samples_path)
        return fid_score

    finally:
        args.sampling_num_samples = original_num_samples


def check_phase1_halting(epochs_no_improve: int, patience: int, epoch: int, training_stage: Stages) -> tuple[bool, int]:
    epochs_no_improve += 1
    print(f"Loss early stopping counter: {epochs_no_improve} out of {patience}")

    if epochs_no_improve >= patience:
        print(f"\n{'='*80}")
        print(f"Phase 1 complete: Loss patience reached after {epoch+1} epochs.")
        print(f"Transitioning to Phase 2: FID-based periodic checkpointing...")
        print(f"{'='*80}\n")
        training_stage.increment()
        return True, epochs_no_improve

    return False, epochs_no_improve


def check_phase2_halting(fid_score: float, best_fid_score: float, fid_no_improve_count: int,
                        epoch: int, training_stage: Stages) -> tuple[bool, float, int]:
    if fid_score < best_fid_score:
        best_fid_score = fid_score
        fid_no_improve_count = 0
        print(f"FID improved! New best: {best_fid_score:.4f}")
        return False, best_fid_score, fid_no_improve_count
    else:
        fid_no_improve_count += 1
        print(f"FID no improvement counter: {fid_no_improve_count} out of 5")

        if fid_no_improve_count >= 5:
            print(f"\n{'='*80}")
            print(f"Phase 2 complete: FID has not improved for 5 consecutive checkpoints.")
            print(f"Halting training at epoch {epoch+1}.")
            print(f"{'='*80}\n")
            training_stage.increment()
            return True, best_fid_score, fid_no_improve_count

        return False, best_fid_score, fid_no_improve_count


def save_best_model(save_path: str, best_fid_model_state, best_fid_model_type: str, best_fid_score: float,
                   best_model_state, best_model_type: str, best_test_loss: float, model: torch.nn.Module):
    if best_fid_model_state is not None:
        torch.save(best_fid_model_state, save_path)
        print(f'Saving Phase 2 best model ({best_fid_model_type} weights, FID: {best_fid_score:.4f})')
    elif best_model_state is not None:
        torch.save(best_model_state, save_path)
        print(f'Saving Phase 1 best model ({best_model_type} weights, loss: {best_test_loss:.4f})')
    else:
        uncompiled_model = getattr(model, "_orig_mod", model)
        torch.save(uncompiled_model.state_dict(), save_path)
        print('Saving the active model (fallback)')


def training_wrapper(args: argparse.Namespace, loss_fn: LossFns, model: torch.nn.Module, data: DataProvider, save_path: str):

    train_dataloader, test_dataloader = data.get_datasets_for_training()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    scaler = torch.amp.GradScaler(device='cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the EMA Model
    ema_decay = 0.9999
    ema_avg_fn = get_ema_multi_avg_fn(ema_decay)
    ema_model = AveragedModel(model, device=device, multi_avg_fn=ema_avg_fn)

    # Setup Phase 1: Loss-based checkpointing
    patience = max(1, int(epochs * args.training_stage1_patience))
    best_test_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    best_model_type = "None"

    # Setup Phase 2: FID-based checkpointing
    checkpoint_interval = max(1, int(epochs * args.training_stage2_patience))
    best_fid_score = float('inf')
    fid_no_improve_count = 0
    best_fid_model_state = None
    best_fid_model_type = "None"

    training_stage = Stages()
    temp_checkpoint_dir = tempfile.mkdtemp()

    try:
        reversal_fns = Reversal(args=args)

        if args.which == 'kac':
            sampler = TorchKacConstantSampler(a=args.kac_a, c=args.kac_c, T=args.T, M=50000, K=4096)
        else:
            sampler = None

        # train the model
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")

            train(args, train_dataloader, model, ema_model, loss_fn, optimizer, scheduler, scaler)

            if training_stage.stage == 'phase1':
                # evaluate both the active model and the ema model
                active_loss = test(args, test_dataloader, model, loss_fn)
                ema_loss = test(args, test_dataloader, ema_model, loss_fn)

                # Gate on the superior configuration for this epoch
                current_best_loss = min(active_loss, ema_loss)

                # Early Stopping Logic (Phase 1) -------------------------------------------------------
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
                    halted, epochs_no_improve = check_phase1_halting(epochs_no_improve, patience, epoch, training_stage)

                    if halted:
                        print(f"\nComputing baseline FID score for Phase 2...")
                        try:
                            baseline_fid = compute_fid_checkpoint(args, ema_model.module, data, reversal_fns, sampler, temp_checkpoint_dir)
                            best_fid_score = baseline_fid
                            print(f"Baseline FID Score: {best_fid_score:.4f}\n")

                            uncompiled_model = getattr(ema_model.module, "_orig_mod", ema_model.module)
                            best_fid_model_state = copy.deepcopy(uncompiled_model.state_dict())
                            best_fid_model_type = "EMA"
                        except Exception as e:
                            print(f"Warning: Baseline FID computation failed: {e}")
                            print("Proceeding to Phase 2 without baseline FID...\n")

            elif training_stage.stage == 'phase2':

                # Phase 2: Periodic FID-based checkpointing
                if (epoch + 1) % checkpoint_interval == 0:
                    print(f"\nPhase 2 checkpoint at epoch {epoch+1}/{epochs}")
                    try:
                        fid_score = compute_fid_checkpoint(args, ema_model.module, data, reversal_fns, sampler, temp_checkpoint_dir)
                        print(f"FID Score (5k samples): {fid_score:.4f}")

                        halted, best_fid_score, fid_no_improve_count = check_phase2_halting(fid_score, best_fid_score, fid_no_improve_count, epoch, training_stage)

                        if fid_no_improve_count == 0:
                            uncompiled_model = getattr(ema_model.module, "_orig_mod", ema_model.module)
                            best_fid_model_state = copy.deepcopy(uncompiled_model.state_dict())
                            best_fid_model_type = "EMA"

                        if halted:
                            break
                    except Exception as e:
                        print(f"Warning: FID computation failed at epoch {epoch+1}: {e}")
                        print("Continuing training without FID checkpoint...")

            if training_stage.stage == 'halt':
                break

            if args.proof_of_concept:
                break

        print("Done!")

        save_best_model(save_path, best_fid_model_state, best_fid_model_type, best_fid_score,
                       best_model_state, best_model_type, best_test_loss, model)

    finally:
        import shutil
        if os.path.exists(temp_checkpoint_dir):
            shutil.rmtree(temp_checkpoint_dir)