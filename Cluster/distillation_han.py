import argparse

import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.optim.lr_scheduler import CosineAnnealingLR

from Cluster.utils.dataHandling import DataProvider
from Cluster.utils.noisifier import Noisify
from Cluster.utils.reversals import Reversal

def distillation_wrapper(args: argparse.Namespace, teacher: torch.nn.Module, student: torch.nn.Module, path: str) -> torch.nn.Module:  

    # Determine device and set up model and loss function accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data: DataProvider = DataProvider(args=args)
    train_dataloader, _ = data.get_datasets_for_training()

    reversal_fns = Reversal(args=args, which='distill')
    noisify_fns = Noisify(args=args)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.distill_lr,
        weight_decay=args.distill_weight_decay
    )
    
    scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.distill_iterations,
            eta_min=1e-6
        )
    
    target_decay = 0.9999
    ema_model = AveragedModel(student, device=device, multi_avg_fn=get_ema_multi_avg_fn(decay=target_decay))

    linspace_of_endpoints = torch.linspace(1, args.time_truncation, args.distill_num_student_steps + 1, dtype=torch.float32, device=device)
    delta_t = (1 - args.time_truncation) / args.distill_num_student_steps

    distill_iter = iter(train_dataloader)
    teacher.eval()
    student.train()
    ema_model.train()
    for iteration in range(args.distill_iterations):
        optimizer.zero_grad()

        # sample a batch from the dataset
        try:
            x_batch, _ = next(distill_iter)
        except StopIteration:
            distill_iter = iter(train_dataloader)
            x_batch, _ = next(distill_iter)
        x_batch = x_batch.to(device)

        # sample a batch of endpoint time steps
        indices = torch.randint(0, args.distill_num_student_steps, (args.training_batch_size,), device=device)
        t_batch = linspace_of_endpoints[indices]

        x_corrupted = noisify_fns.noisify(x0=x_batch, t=t_batch)

        x_teacher = reversal_fns.integrator(
            model=teacher,
            x_batch=x_corrupted,
            t_start=t_batch,
            dt=delta_t,
            num_substeps=args.distill_num_teacher_substeps,
        )

        x_student = reversal_fns.student_integrate(
            model=student,
            x_batch=x_corrupted,
            t_batch=t_batch,
            dt=delta_t,
        )

        loss = torch.nn.functional.mse_loss(x_student, x_teacher) / (delta_t**2)
        if iteration % 1000 == 0:
            print(f'iteration  {iteration}')
        
        # 4. Optimization step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        ema_model.update_parameters(student)

    uncompiled_model = getattr(student, "_orig_mod", student)
    torch.save(uncompiled_model.state_dict(), f'{path}student_{args.distill_num_student_steps}.pth')
    print(f"saved best loss model to:  {f'{path}student_{args.distill_num_student_steps}.pth'}")

    uncompiled_model = getattr(ema_model.module, "_orig_mod", ema_model.module)
    torch.save(uncompiled_model.state_dict(), f'{path}student_{args.distill_num_student_steps}_ema.pth')
    print(f"saved best loss model to:  {f'{path}student_{args.distill_num_student_steps}_ema.pth'}")

    return student