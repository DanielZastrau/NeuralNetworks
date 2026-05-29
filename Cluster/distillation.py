import argparse

import torch
import torch.nn as nn

from Cluster.neuralNetworkSmall import ConditionalUNet
from Cluster.neuralNetworkOpenAI import UNetModel

from Cluster.utils.diffusion import f, g, b
from Cluster.utils.dataHandling import DataProvider
from Cluster.utils.noisifier import Noisify
from Cluster.utils.modelGetter import model_getter

def teacher_integrate(model: nn.Module, x_batch: torch.Tensor, t_batch: torch.Tensor, delta_t: float, num_substeps: int) -> torch.Tensor:
    """Integrates the teacher over [t*, t* - delta_t] using num_substeps many uniform substeps

    This is the Euler-Maruyama Scheme also used to solve the SDE formulation of the reverse process.

    TODO: This duplicates code from the other SDE sampling file. Should fix this.
    """
    device = next(model.parameters()).device

    dt_sub = delta_t / num_substeps
    x_star = x_batch.clone()
    t_curr = t_batch.clone()
    
    with torch.no_grad():
        for _ in range(num_substeps):

            # Get continuous coefficients
            f_t_x = f(t_curr, x_star)
            g_t = g(t_curr).view(-1, 1, 1, 1)
            b_t = b(t_curr).view(-1, 1, 1, 1)

            # Predict score using continuous time
            pred_noise = model(x_star, t_curr)
            pred_score = - pred_noise / torch.sqrt(1 - b_t**2)

            # 2. Scale updates explicitly by dt and sqrt(dt)
            drift_update = f_t_x * dt_sub
            score_update = (g_t ** 2) * pred_score * dt_sub
            noise_injection = g_t * torch.sqrt(torch.tensor(dt_sub, device=device)) * torch.randn_like(x_star)
            
            # Continuous SDE reverse step formula
            x_star = x_star - drift_update + score_update + noise_injection
            t_curr -= dt_sub

    return x_star


def student_integrate(model: nn.Module, x_batch: torch.Tensor, t_batch: torch.Tensor, delta_t: float) -> torch.Tensor:
    """Integrates the student over [t*, t* - delta_t]
    
    In the future I want to implement other student integrators. SDE Euler-Maruyama, ODE RK45"""

    # TODO implement another integrator
    # * explicit euler step as proposed in the algorithm by "Han et al 2025 - DistillKac: Few Step Image Generation via Damped Wave Equations"
    return x_batch - model(x_batch, t_batch) * delta_t


def distillation_wrapper(args: argparse.Namespace, save_path: str, model_path: str = ''):
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


    print('\nInitialize the teacher.')
    if args.where == 'local':
        teacher = ConditionalUNet(in_channels=data.data_dims.channels, out_channels=data.data_dims.channels).to(device)
    else:
        print('getting the large model teacher')
        teacher = model_getter(args=args).to(device)
    teacher.load_state_dict(torch.load(path, map_location=device))


    print('\nInitialize the student.')
    if args.where == 'local':
        student = ConditionalUNet(in_channels=data.data_dims.channels, out_channels=data.data_dims.channels).to(device)
    else:
        print('getting the large model student')
        student = model_getter(args=args).to(device)
    student.load_state_dict(torch.load(path, map_location=device))


    print('\nGot the noise')
    noisifier = Noisify(args=args).noisify


    print('\nSet optimizer to AdamW')
    optimizer = torch.optim.AdamW(student.parameters(), lr=2e-4)


    print('\nPreparing distillation\n')
    # Number of teacher substeps, i.e. distilling N teacher steps into 1 student step
    num_substeps = args.num_teacher_substeps

    # Number of student steps, i.e. in the end we want to sample with M steps
    num_steps = args.num_student_steps

    # !This needs to match the training setup
    # TODO: Since there are dependencies across functionalities, this should be outsourced to a higher hierarchy level from where it can be passed to everything below
    # * 1e-5 as specified by "Song et al 2021 - Score based generative modelling through sdes" and as referenced by "Duong Chemseddine 2025 - Telegraphers Generative Model via Kac Flows"
    eps = 1e-3
    linspace_of_endpoints = torch.linspace(1, eps, num_steps, dtype=torch.float32)
    delta_t = 1 / num_steps

    teacher.eval()
    student.train()
    for iteration in range(args.iterations):
        print(f'Starting iteration  {iteration}')
        optimizer.zero_grad()

        # sample a batch from the dataset
        x_batch, _ = next(iter(train_dataloader))

        # sample a batch of endpoint time steps
        indices = torch.randint(0, num_steps, (args.batch_size,))
        t_batch = linspace_of_endpoints[indices]

        # noisify x_batch according to t_batch
        x_batch_corrupted = noisifier(x0=x_batch, t=t_batch)

        # integrate backwards in time using the teacher method and N uniform substeps
        x_target = teacher_integrate(
            model=teacher, x_batch=x_batch_corrupted,
            t_batch=t_batch, delta_t=delta_t,
            num_substeps=num_substeps
        )

        # integrate backwards in time using the student method and 1 substep
        x_calc = student_integrate(
            model=student, x_batch=x_batch_corrupted,
            t_batch=t_batch, delta_t=delta_t,
        )

        # compute the loss and update the weights
        loss = nn.functional.mse_loss(x_target, x_calc)
        print(f'---loss  {loss.item()}---\n')
        
        # 4. Optimization step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

    torch.save(student.state_dict(), save_path)