import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets    # type: ignore
from torchvision.transforms import v2    # type: ignore
from torch.utils.data import DataLoader

from neuralNetworkSmall import ConditionalUNet

from utils.diffusion import f, g, b, noisify

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

    # For now do the explicit euler step as proposed in the algorithm by "Han et al 2025 - DistillKac: Few Step Image Generation via Damped Wave Equations"
    return x_batch - model(x_batch, t_batch) * delta_t


def distillation_wrapper(args: argparse.Namespace):
    """Wraps together the functions and boilerplate"""    

    # Determine device and set up model and loss function accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print('\nInitialize the teacher.')
    teacher = ConditionalUNet(in_channels=3, out_channels=3).to(device)
    teacher.load_state_dict(torch.load(args.model, map_location=device))


    print('\nInitialize the student.')
    student = ConditionalUNet(in_channels=3, out_channels=3).to(device)
    student.load_state_dict(torch.load(args.model, map_location=device))


    print('\nLoad and transform the dataset')
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),  # Scales to [0, 1]
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Shifts to [-1, 1]
    ])

    training_data = datasets.CIFAR10(    # type: ignore
        root="../data",
        train=True,
        download=True,
        transform=transform
    )

    train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)    # type: ignore


    print('\nSet optimizer to AdamW')
    optimizer = torch.optim.AdamW(student.parameters(), lr=2e-4)


    print('\nPreparing distillation\n')
    # Number of teacher substeps, i.e. distilling N teacher steps into 1 student step
    num_substeps = 2

    # Number of student steps, i.e. in the end we want to sample with 4096 steps which is half of what I use for the diffusion teacher sde
    num_steps = 4096
    eps = 1e-5    # 1e-5 as specified by "Song et al 2021 - Score based generative modelling through sdes" and as referenced by "Duong Chemseddine 2025 - Telegraphers Generative Model via Kac Flows"
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
        indices = torch.randint(0, num_steps, (128,))
        t_batch = linspace_of_endpoints[indices]

        # noisify x_batch according to t_batch
        x_batch_corrupted = noisify(x_0=x_batch, t=t_batch)

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

    student._save_to_state_dict(f'{args.model[:-4]}_student.pth')

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--iterations', type=int, required=True)

    args = parser.parse_args()

    distillation_wrapper(args=args)