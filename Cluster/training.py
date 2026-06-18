import argparse
import copy
import os

import torch
import torch_fidelity
from torch.optim.lr_scheduler import LinearLR, ConstantLR, SequentialLR, CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from Cluster.utils.dataHandling import DataProvider
from Cluster.utils.lossFunctions import LossFns
from Cluster.utils.stages import Stages
from Cluster.utils.sample_kac import TorchKacConstantSampler
from Cluster.utils.reversals import Reversal
from Cluster.sampling import sample, sample_wrapper
from Cluster.utils.uint8_utils import Uint8Dataset, to_uint8_rgb

corruption_counter = 0

def train(args: argparse.Namespace, x_batch: torch.Tensor,
          model: torch.nn.Module, ema_model: AveragedModel,
          loss_fn: LossFns, optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler.LRScheduler):
    
    global corruption_counter

    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)

    model.train()
    ema_model.train()

    x_batch = x_batch.to(device)
    optimizer.zero_grad()

    loss = loss_fn.loss(model=model, mini_batch=x_batch)
    if args.training_verbosity == 'verbose':
        print(f'Loss  {loss.item()}')

    # Scales the loss and completes the backward pass
    loss.backward()

    # Capture and log the norm before clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    if args.training_verbosity == 'verbose':
        print(f'Grad Norm: {grad_norm.item()}')
        if not torch.isfinite(grad_norm):
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    print(f"NaN/Inf gradient in layer: {name}")

            # if there was an invalid training step 10 times in a row stop
            if corruption_counter != 10:
                corruption_counter += 1
            else:
                raise RuntimeError('Encountered corruption. Stopping the training.')
        else:
            corruption_counter = 0
    
    # Step optimizer and scaler
    optimizer.step()
    scheduler.step()
    ema_model.update_parameters(model)
    
    if args.training_verbosity == 'verbose':
        print(f'Learning rate:  {scheduler.get_last_lr()}')


def test(args: argparse.Namespace, dataloader, model: torch.nn.Module, loss_fn: LossFns) -> float:    # type: ignore    due to type of dataloader partially unknown warning

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_batches = len(dataloader)

    model.eval()
    loss = 0
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)

            loss += loss_fn.loss(model=model, mini_batch=X).detach()
            if args.training_verbosity == 'verbose':
                print(f'loss  {loss.item()}')

            if args.proof_of_concept:
                break

    avg_test_loss = loss.item() / num_batches

    return avg_test_loss


def training_wrapper(args: argparse.Namespace, loss_fn: LossFns,
                     reversal_fns: Reversal, model: torch.nn.Module,
                     data: DataProvider, save_path: str):

    train_dataloader, test_dataloader = data.get_datasets_for_training()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Initialize optimizer with the TARGET scaled learning rate
    target_lr = args.lr

    optimizer = torch.optim.AdamW(model.parameters(), lr=target_lr, weight_decay=args.training_optimizer_weight_decay)

    if not args.model:    # ! I.e. if a new model is trained 
        # Define warm up steps, e.g. 400k * 0.05 = 4k * 0.05 = 20k
        warmup_steps = int(args.training_iterations * 0.05)

        # Warmup: Linearly increase LR from near-zero (target_lr * 1e-8) to target_lr
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        print(f'Using warmup and cosine annealing learning rate scheduler.')
        cosine_annealing = CosineAnnealingLR(
            optimizer,
            T_max=(args.training_iterations - warmup_steps),
            eta_min=1e-6
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_annealing],
            milestones=[warmup_steps]
            )

    else:    # ! I.e. if a pre-trained model is passed
        # Use a constant learning rate for finetuning/pretrained model
        scheduler = ConstantLR(
            optimizer,
            factor=0.5,
            total_iters=1
        )

    # Initialize the EMA Model with dynamic warmup
    target_decay = 0.9999
    ema_model = AveragedModel(model, device=device, multi_avg_fn=get_ema_multi_avg_fn(decay=target_decay))

    # Setup periodic checkpointing wrt loss
    best_loss = float('inf')
    loss_save_path = ''

    # Setup periodic checkpointing wrt fid
    best_score = float('inf')
    score_save_path = ''

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
        if iteration % args.training_logging_period == 0 and args.training_verbosity in ['normal', 'verbose']:
            print(f'iteration  {iteration}----------------------------')

        # the iterator does not need to be initialized every time
        try:
            x_batch, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            x_batch, _ = next(train_iter)

        train(args=args, x_batch=x_batch, model=model,
                ema_model=ema_model, loss_fn=loss_fn,
                optimizer=optimizer, scheduler=scheduler)


        # =========================================================================================
        # ! periodic checking
        # =========================================================================================


        # regularly sample a small grid to check progress
        if (iteration + 1) % args.training_evaluation_period_grid == 0:
            tmp_mode = args.sampling_mode
            tmp_num_samples = args.sampling_num_samples
            tmp_num_steps = args.sampling_num_steps
            args.sampling_mode = '8x8'
            args.sampling_num_samples = 64
            args.sampling_num_steps = args.training_evaluation_period_grid_num_steps

            grid_save_path = f'samples8x8_{args.which}_{iteration+1}.png'
            if args.where == 'cluster':
                if not os.path.exists(f'/work/zastrau/samples/{args.which}'):
                    os.mkdir(f'/work/zastrau/samples/{args.which}')
                grid_save_path = f'/work/zastrau/samples/{args.which}/{grid_save_path}'
            else:    # args.where == 'local':
                if not os.path.exists(f'./samples/{args.which}'):
                    os.mkdir(f'./samples/{args.which}')
                grid_save_path = f'./samples/{args.which}/{grid_save_path}'

            ema_model.eval()
            sample_wrapper(
                args=args,
                model=ema_model,
                data=data,
                sampler=sampler,
                reversal_fns=reversal_fns,
                save_path=grid_save_path,
            )

            args.sampling_mode = tmp_mode
            args.sampling_num_samples = tmp_num_samples
            args.sampling_num_steps = tmp_num_steps

            print(f'-----------------------------------------------generated an 8x8 grid and saved it to:  {grid_save_path}')


        if not args.training_use_early_halting:
            if (iteration + 1) % args.training_evaluation_period_loss == 0:

                ema_loss = test(args, test_dataloader, ema_model, loss_fn)
                if args.training_verbosity == 'verbose':
                    print(f'Tested the ema model. Loss    {ema_loss}.')

                if ema_loss < best_loss:
                    best_loss = ema_loss

                    # clean up last checkpoint,
                    if loss_save_path:
                        os.remove(loss_save_path)
                    loss_save_path = f'/work/zastrau/{args.which}_iteration{iteration}_loss{ema_loss:.8f}.pth'

                    uncompiled_model = getattr(ema_model.module, "_orig_mod", ema_model.module)
                    torch.save(best_model_state, loss_save_path)
                    print(f"saved best loss model to:  {loss_save_path},    loss {ema_loss}")

            if (iteration + 1) % args.training_evaluation_period_fid == 0:

                ema_model.eval()
                samples = sample(
                    args=args,
                    model=ema_model,
                    data=data,
                    sampler=sampler,
                    num_samples=args.training_evaluation_period_fid_num_samples,
                    num_steps=args.training_evaluation_period_fid_num_steps,
                    reversal_fns=reversal_fns
                )
                generated_ds = Uint8Dataset(to_uint8_rgb(samples).detach().cpu())

                metrics = torch_fidelity.calculate_metrics(
                    input1=real_ds,
                    input2=generated_ds,
                    batch_size=256,
                    fid=True,
                    cuda=(('cuda' if torch.cuda.is_available() else 'cpu') == 'cuda'),
                    verbose=False,
                )
                ema_score = metrics['frechet_inception_distance']
                if args.training_verbosity == 'verbose':
                    print(f"Tested the ema model. FID Score ({args.training_evaluation_period_fid_num_samples} samples): {ema_score:.4f}")

                if ema_score < best_score:
                    best_score = ema_score

                    # clean up last checkpoint,
                    if score_save_path:
                        os.remove(score_save_path)
                    score_save_path = f'/work/zastrau/{args.which}_iteration{iteration}_score{ema_score:.2f}.pth'

                    uncompiled_model = getattr(ema_model.module, "_orig_mod", ema_model.module)
                    torch.save(best_model_state, score_save_path)
                    print(f"saved best score model to:  {score_save_path},    loss {ema_score}")


        # =========================================================================================
        # ! Early halting.
        # =========================================================================================


        if args.training_use_early_halting:
            # evaluate the model periodically on the loss function (computationally faster than generating samples)
            if training_stage.stage == 'phase1' and (iteration + 1) % args.training_stage1_period == 0:
                
                # evaluate just the ema model
                ema_loss = test(args, test_dataloader, ema_model, loss_fn)
                if args.training_verbosity == 'verbose':
                    print(f'tested the ema model  --  loss    {ema_loss}')

                # Early Stopping Logic (Phase 1) -------------------------------------------------------
                if ema_loss < best_test_loss:
                    best_test_loss = ema_loss
                    counter_no_improve = 0

                    # Extract and store the state dict of the superior model
                    uncompiled_model = getattr(ema_model.module, "_orig_mod", ema_model.module)
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