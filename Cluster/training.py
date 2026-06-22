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


def training_wrapper(args: argparse.Namespace, loss_fn: LossFns, model: torch.nn.Module, data: DataProvider):

    train_dataloader, test_dataloader = data.get_datasets_for_training()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the EMA Model with dynamic warmup
    target_decay = 0.9999
    ema_model = AveragedModel(model, device=device, multi_avg_fn=get_ema_multi_avg_fn(decay=target_decay))

    # Initialize optimizer with the TARGET scaled learning rate
    if args.training_optimizer == 'adamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.training_optimizer_weight_decay)
    elif args.training_optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.training_scheduler == 'cosine': 
        print(f'Using warmup and cosine annealing learning rate scheduler.')
        # Define warm up steps, e.g. 400k * 0.05 = 4k * 0.05 = 20k
        warmup_steps = int(args.training_iterations * 0.05)

        # Warmup: Linearly increase LR from near-zero (target_lr * 1e-8) to target_lr
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_steps
        )

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

    elif args.training_scheduler == 'constant':
        scheduler = ConstantLR(
            optimizer,
            factor=1,
            total_iters=1
        )

    # Setup periodic sampling of a grid
    grid_reversal_fn = Reversal(args=args, which='grid')

    # Setup periodic checkpointing wrt loss
    best_loss: float = float('inf')
    loss_save_path = ''

    # Setup periodic checkpointing wrt fid
    score_reversal_fn = Reversal(args=args, which='fid')
    best_score: float = float('inf')
    score_save_path = ''

    # --- Setup for periodic FID during training ---
    real_ds = data.get_dataset_for_periodic_eval()

    # initiate the sampler if needed    
    if args.which == 'kac':
        sampler = TorchKacConstantSampler(a=args.kac_a, c=args.kac_c, T=args.T, M=50_000, K=4_096)
    else:
        sampler = None

    # train the model
    train_iter = iter(train_dataloader)
    for iteration in range(args.training_iterations):
        if iteration % args.training_logging_period == 0 and args.training_verbosity in ['normal', 'verbose']:
            print(f'iteration  {iteration}----------------------------')

        # the iterator does not need to be initialized every time
        try:
            x_batch, _ = next(train_iter)
            x_batch = x_batch.to(device, dtype=torch.float32)
        except StopIteration:
            train_iter = iter(train_dataloader)
            x_batch, _ = next(train_iter)
            x_batch = x_batch.to(device, dtype=torch.float32)


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
                reversal_fns=grid_reversal_fn,
                save_path=grid_save_path,
            )

            args.sampling_mode = tmp_mode
            args.sampling_num_samples = tmp_num_samples
            args.sampling_num_steps = tmp_num_steps

            print(f'-----------------------------------------------generated an 8x8 grid and saved it to:  {grid_save_path}')


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
                torch.save(uncompiled_model.state_dict(), loss_save_path)
                print(f"saved best loss model to:  {loss_save_path},    loss {ema_loss}")

        if (iteration + 1) % args.training_evaluation_period_fid == 0:

            tmp_samples = args.sampling_num_samples
            tmp_num_steps = args.sampling_num_steps
            args.sampling_num_samples = args.training_evaluation_period_fid_num_samples
            args.sampling_num_steps = args.training_evaluation_period_fid_num_steps

            ema_model.eval()
            samples = sample(
                args=args,
                model=ema_model,
                data=data,
                sampler=sampler,
                reversal_fns=score_reversal_fn
            )
            generated_ds = Uint8Dataset(to_uint8_rgb(samples, data).detach().cpu())

            metrics = torch_fidelity.calculate_metrics(
                input1=real_ds,
                input2=generated_ds,
                batch_size=256,
                fid=True,
                cuda=(('cuda' if torch.cuda.is_available() else 'cpu') == 'cuda'),
                verbose=False,
            )
            ema_score = metrics['frechet_inception_distance']

            args.sampling_num_samples = tmp_samples
            args.sampling_num_steps = tmp_num_steps

            if args.training_verbosity == 'verbose':
                print(f"Tested the ema model. FID Score ({args.training_evaluation_period_fid_num_samples} samples): {ema_score:.4f}")

            if ema_score < best_score:
                best_score = ema_score

                # clean up last checkpoint,
                if score_save_path:
                    os.remove(score_save_path)
                score_save_path = f'/work/zastrau/{args.which}_iteration{iteration}_score{ema_score:.2f}.pth'

                uncompiled_model = getattr(ema_model.module, "_orig_mod", ema_model.module)
                torch.save(uncompiled_model.state_dict(), score_save_path)
                print(f"saved best score model to:  {score_save_path},    loss {ema_score}")

    print("Done!")