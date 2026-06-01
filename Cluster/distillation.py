import argparse

import torch
import torch.nn as nn

from Cluster.networks.neuralNetworkSmall import ConditionalUNet
from Cluster.utils.modelGetter import model_getter

from Cluster.utils.dataHandling import DataProvider
from Cluster.utils.noisifier import Noisify
from Cluster.utils.reversals import Reversal

def distillation_wrapper(args: argparse.Namespace, save_path: str, reversal_fns: Reversal, model_path: str = '') -> torch.nn.Module:
    """Wraps together the functions and boilerplate"""    

    # Determine device and set up model and loss function accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('\nInitialized the data')
    data: DataProvider = DataProvider(args=args)
    train_dataloader, _ = data.get_datasets_for_training()


    if args.what == 'full':
        path = model_path
    else:
        path = args.model
    print(f'\nInstantiating the models from {path}')


    print(f'\nLoading the state dict')
    state_dict = torch.load(path, map_location=device)


    print('\nInitializing the teacher.')
    if args.where == 'local':
        teacher = ConditionalUNet(in_channels=data.data_dims.channels, out_channels=data.data_dims.channels).to(device)
    else:
        print('getting the large model teacher')
        teacher = model_getter(args=args).to(device)
    teacher.load_state_dict(state_dict)
    teacher.eval()
    if args.where == 'cluster':
        teacher = torch.compile(teacher)


    print('\nInitializing the student.')
    if args.where == 'local':
        student = ConditionalUNet(in_channels=data.data_dims.channels, out_channels=data.data_dims.channels).to(device)
    else:
        print('getting the large model student')
        student = model_getter(args=args).to(device)
    student.load_state_dict(state_dict)
    student.train()
    if args.where == 'cluster':
        student = torch.compile(student)


    print('\nGot the noise')
    noisifier = Noisify(args=args).noisify


    print('\nSet optimizer to AdamW')
    optimizer = torch.optim.AdamW(student.parameters(), lr=2e-4)


    print('\nPreparing distillation\n')
    # Number of teacher substeps, i.e. distilling N teacher steps into 1 student step
    num_substeps = args.num_teacher_substeps

    # Number of student steps, i.e. in the end we want to sample with M steps
    num_steps = args.num_student_steps

    if args.which == 'diffusion':
        # * 1e-5 as specified by "Song et al 2021 - Score based generative modelling through sdes" and as referenced by "Duong Chemseddine 2025 - Telegraphers Generative Model via Kac Flows"
        eps = args.time_truncation
    else:
        eps = 0
    
    linspace_of_endpoints = torch.linspace(1, eps, num_steps, dtype=torch.float32, device=device)
    delta_t = 1 / num_steps


    for iteration in range(args.iterations):
        print(f'Starting iteration  {iteration}')
        optimizer.zero_grad()

        # sample a batch from the dataset
        x_batch, _ = next(iter(train_dataloader))
        x_batch = x_batch.to(device)

        # sample a batch of endpoint time steps
        indices = torch.randint(0, num_steps, (args.batch_size,), device=device)
        t_batch = linspace_of_endpoints[indices]

        # noisify x_batch according to t_batch
        # TODO still misses MMD and Schrödinger
        x_batch_corrupted = noisifier(x0=x_batch, t=t_batch)

        # integrate backwards in time using the teacher method and N uniform substeps
        # TODO still misses Kac, MMD, Schrödinger
        x_target = reversal_fns.teacher_integrate(
            model=teacher,
            x_batch=x_batch_corrupted,
            t_start=t_batch,
            dt=delta_t,
            num_substeps=num_substeps,
        )

        # integrate backwards in time using the student method and 1 substep
        x_calc = reversal_fns.student_integrate(
            model=student,
            x_batch=x_batch_corrupted,
            t_batch=t_batch,
            dt=delta_t
        )

        print(f'DEBUGGING: x_target  {x_target.mean()}    x_calc {x_calc.mean()}')

        # compute the loss and update the weights
        loss = nn.functional.mse_loss(x_target, x_calc)
        print(f'---loss  {loss.item()}---\n')
        
        # 4. Optimization step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

    torch.save(student.state_dict(), save_path)

    return student