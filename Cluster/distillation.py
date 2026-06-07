import argparse

import torch
import torch.nn as nn

from Cluster.networks.neuralNetworkSmall import ConditionalUNet
from Cluster.utils.modelGetter import model_getter

from Cluster.utils.dataHandling import DataProvider
from Cluster.utils.noisifier import Noisify
from Cluster.utils.reversals import Reversal

def distillation_wrapper(args: argparse.Namespace, save_path: str, reversal_fns: Reversal,
                         noisify_fns: Noisify, model_path: str = '') -> torch.nn.Module:
    """Wraps together the functions and boilerplate"""    

    # Determine device and set up model and loss function accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data: DataProvider = DataProvider(args=args)
    train_dataloader, _ = data.get_datasets_for_training()


    if args.what == 'full':
        path = model_path
    else:
        path = args.model
    state_dict = torch.load(path, map_location=device)

    if args.where == 'local':
        teacher = ConditionalUNet(in_channels=data.data_dims.channels, out_channels=data.data_dims.channels).to(device)
    else:
        teacher = model_getter(args=args).to(device)
    teacher.load_state_dict(state_dict)
    teacher.eval()
    if args.where == 'cluster':
        teacher = torch.compile(teacher)


    if args.where == 'local':
        student = ConditionalUNet(in_channels=data.data_dims.channels, out_channels=data.data_dims.channels).to(device)
    else:
        student = model_getter(args=args).to(device)
    student.load_state_dict(state_dict)
    student.train()
    if args.where == 'cluster':
        student = torch.compile(student)


    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.distill_lr
    )

    if args.which == 'diffusion':
        # * 1e-5 as specified by "Song et al 2021 - Score based generative modelling through sdes" and as referenced by "Duong Chemseddine 2025 - Telegraphers Generative Model via Kac Flows"
        eps = args.time_truncation
    else:
        eps = 0
    
    linspace_of_endpoints = torch.linspace(1, eps, args.distill_num_student_steps, dtype=torch.float32, device=device)
    delta_t = 1 / args.distill_num_student_steps

    distill_iter = iter(train_dataloader)
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

        # noisify x_batch according to t_batch
        # TODO still misses Schrödinger
        x_batch_corrupted = noisify_fns.noisify(x0=x_batch, t=t_batch)

        # integrate backwards in time using the teacher method and N uniform substeps
        # TODO still misses Schrödinger
        x_target = reversal_fns.teacher_integrate(
            model=teacher,
            x_batch=x_batch_corrupted,
            t_start=t_batch,
            dt=delta_t,
            num_substeps=args.distill_num_teacher_substeps,
        )

        # integrate backwards in time using the student method and 1 substep
        x_calc = reversal_fns.student_integrate(
            model=student,
            x_batch=x_batch_corrupted,
            t_batch=t_batch,
            dt=delta_t
        )

        # compute the loss and update the weights
        loss = nn.functional.mse_loss(x_target, x_calc)
        if iteration % 1000 == 0:
            print(f'iteration  {iteration}')
        
        # 4. Optimization step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

    torch.save(student.state_dict(), save_path)

    return student