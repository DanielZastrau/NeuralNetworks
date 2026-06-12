import argparse
import copy
import os

import torch
import torch_fidelity
from torch.optim.lr_scheduler import LinearLR, ConstantLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from Cluster.utils.dataHandling import DataProvider
from Cluster.utils.lossFunctions import LossFns
from Cluster.utils.stages import Stages
from Cluster.utils.sample_kac import TorchKacConstantSampler
from Cluster.utils.reversals import Reversal
from Cluster.sampling import sample, sample_wrapper
from Cluster.utils.uint8_utils import Uint8Dataset, to_uint8_rgb

def train(x_batch: torch.Tensor,
          model: torch.nn.Module, ema_model: AveragedModel,
          loss_fn: LossFns, optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler.LRScheduler, scaler: torch.amp.GradScaler):
    
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)

    model.train()
    ema_model.train()

    x_batch = x_batch.to(device)
    optimizer.zero_grad()

    loss = loss_fn.loss(model=model, mini_batch=x_batch)

    # Scales the loss and completes the backward pass
    scaler.scale(loss).backward()

    # Unscale gradients before clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Step optimizer and scaler
    scale_before = scaler.get_scale()
    
    # Step optimizer and scaler
    scaler.step(optimizer)
    scaler.update()
    
    # Only step the scheduler if the optimizer actually updated the weights
    if scaler.get_scale() >= scale_before:
        scheduler.step()

    # Update EMA model parameters
    ema_model.update_parameters(model)


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


def training_wrapper(args: argparse.Namespace, loss_fn: LossFns,
                     reversal_fns: Reversal, model: torch.nn.Module,
                     data: DataProvider, save_path: str):

    train_dataloader, test_dataloader = data.get_datasets_for_training()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define step counts
    warmup_steps = int(args.training_iterations * 0.05)

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

    # Initialize the EMA Model with dynamic warmup
    target_decay = 0.9999
    ema_model = AveragedModel(model, device=device, multi_avg_fn=get_ema_multi_avg_fn(decay=target_decay))

    # Setup Phase 1: Loss-based checkpointing
    best_test_loss = float('inf')
    counter_no_improve = 0
    best_model_state = None
    best_model_type = "None"

    # Setup Phase 2: FID-based checkpointing
    best_fid_score = float('inf')
    fid_no_improve_count = 0
    best_fid_model_state = None
    best_fid_model_type = "None"

    # --- Setup for periodic FID during training ---
    real_ds = data.get_dataset_for_periodic_eval()

    # initiate the sampler if needed    
    if args.which == 'kac':
        sampler = TorchKacConstantSampler(a=args.kac_a, c=args.kac_c, T=args.T, M=50_000, K=4_096)
    else:
        sampler = None


    # train the model
    training_stage = Stages()
    train_iter = iter(train_dataloader)
    for iteration in range(args.training_iterations):
        if iteration % 1000 == 0:
            print(f'iteration  {iteration}----------------------------')

        # the iterator does not need to be initialized every time
        try:
            x_batch, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            x_batch, _ = next(train_iter)

        train(x_batch=x_batch, model=model,
                ema_model=ema_model, loss_fn=loss_fn,
                optimizer=optimizer, scheduler=scheduler,
                scaler=scaler)


        # every 5k iterations sample a small grid to check progress
        if (iteration + 1) % 5_000 == 0:
            tmp_mode = args.sampling_mode
            tmp_num = args.sampling_num_samples
            args.sampling_mode = '8x8'
            args.sampling_num_samples = 64

            tmp_save_path = f'samples8x8_{args.which}_{iteration+1}.png'
            if args.where == 'cluster':
                if not os.path.exists('/work/zastrau/samples'):
                    os.mkdir(f'/work/zastrau/samples{args.which}')
                tmp_save_path = f'/work/zastrau/samples/{tmp_save_path}'
            else:    # args.where == 'local':
                if not os.path.exists(f'./samples'):
                    os.mkdir('./samples')
                tmp_save_path = f'./samples/{tmp_save_path}'

            ema_model.eval()
            sample_wrapper(
                args=args,
                model=ema_model,
                data=data,
                sampler=sampler,
                reversal_fns=reversal_fns,
                save_path=tmp_save_path,
            )

            args.sampling_mode = tmp_mode
            args.sampling_num_samples = tmp_num

            print(f'-----------------------------------------------generated an 8x8 grid and saved it to:  {tmp_save_path}')


        # evluate the model periodically on the loss function (computationally faster than generating samples)
        if training_stage.stage == 'phase1' and (iteration + 1) % args.training_stage1_period == 0:
            
            # evaluate just the ema model
            ema_loss = test(args, test_dataloader, ema_model, loss_fn)

            # Early Stopping Logic (Phase 1) -------------------------------------------------------
            if ema_loss < best_test_loss:
                best_test_loss = ema_loss
                counter_no_improve = 0

                # Extract and store the state dict of the superior model
                uncompiled_model = getattr(ema_model.module, "_orig_mod", ema_model.module)
                best_model_state = copy.deepcopy(uncompiled_model.state_dict())
                best_model_type = "EMA"
                torch.save(best_model_state, save_path)
                print(f"saved best model to:  {save_path},    loss {ema_loss}")
            else:
                counter_no_improve += 1
                print(f"Loss early stopping counter: {counter_no_improve} out of {args.training_stage1_patience}")

                if counter_no_improve >= args.training_stage1_patience:
                    print(f"\n{'='*80}")
                    print(f"Phase 1 complete: Loss patience reached after {iteration+1} iterations.")
                    print(f"Transitioning to Phase 2: FID-based periodic checkpointing...")
                    print(f"{'='*80}\n")
                    training_stage.increment()

                    print(f"\nComputing baseline FID score for Phase 2...")
                    # sample images from the ema model to calculate a reduced fid score
                    ema_model.eval()
                    samples = sample(
                        args=args,
                        model=ema_model,
                        data=data,
                        sampler=sampler,
                        num_samples=args.training_stage2_samples,
                        num_steps=args.training_stage2_num_steps,
                        reversal_fns=reversal_fns
                    )
                    generated_ds = Uint8Dataset(to_uint8_rgb(samples).cpu())

                    # calculate the fid score
                    metrics = torch_fidelity.calculate_metrics(
                        input1=real_ds,
                        input2=generated_ds,
                        batch_size=256,
                        fid=True,
                        cuda=(('cuda' if torch.cuda.is_available() else 'cpu') == 'cuda'),
                        verbose=False,
                    )
                    baseline_fid_score = metrics['frechet_inception_distance']
                    best_fid_score = baseline_fid_score
                    print(f"Baseline FID Score: {best_fid_score:.4f}\n")

                    # copy the current model
                    uncompiled_model = getattr(ema_model.module, "_orig_mod", ema_model.module)
                    best_fid_model_state = copy.deepcopy(uncompiled_model.state_dict())
                    best_fid_model_type = "EMA"
                    torch.save(best_fid_model_state, save_path)
                    print(f"saved best model to:  {save_path}")


        # If loss patience has already been reached, evaluate the model periodically on the fid score for small sample size
        elif training_stage.stage == 'phase2' and (iteration + 1) % args.training_stage2_period == 0:
                
            print(f"\nPhase 2 checkpoint at epoch {iteration+1}/{args.training_iterations}")
            # sample images from the ema model to calculate a reduced fid score
            ema_model.eval()
            samples = sample(
                args=args,
                model=ema_model,
                data=data,
                sampler=sampler,
                num_samples=args.training_stage2_samples,
                num_steps=args.training_stage2_num_steps,
                reversal_fns=reversal_fns
            )
            generated_ds = Uint8Dataset(to_uint8_rgb(samples).detach().cpu())

            # calculate the fid score
            metrics = torch_fidelity.calculate_metrics(
                input1=real_ds,
                input2=generated_ds,
                batch_size=256,
                fid=True,
                cuda=(('cuda' if torch.cuda.is_available() else 'cpu') == 'cuda'),
                verbose=False,
            )
            fid_score = metrics['frechet_inception_distance']

            print(f"FID Score ({args.training_stage2_samples} samples): {fid_score:.4f}")

            if fid_score < best_fid_score:
                best_fid_score = fid_score
                fid_no_improve_count = 0
                print(f"FID improved! New best: {best_fid_score:.4f}")

                uncompiled_model = getattr(ema_model.module, "_orig_mod", ema_model.module)
                best_fid_model_state = copy.deepcopy(uncompiled_model.state_dict())
                best_fid_model_type = "EMA"
                torch.save(best_fid_model_state, save_path)
                print(f"saved best fid model to:  {save_path}")

            else:
                fid_no_improve_count += 1
                print(f"FID no improvement counter: {fid_no_improve_count} out of {args.training_stage2_patience}")

                if fid_no_improve_count >= args.training_stage2_patience:
                    print(f"\n{'='*80}")
                    print(f"Phase 2 complete: FID has not improved for {args.training_stage2_patience} consecutive checkpoints.")
                    print(f"Halting training at iteration {iteration+1}.")
                    print(f"{'='*80}\n")
                    training_stage.increment()

        # If loss and fid patience have been reached, halt
        if training_stage.stage == 'halt':
            break

    print("Done!")

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