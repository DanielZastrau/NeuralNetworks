import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets    # type: ignore
from torchvision.transforms import v2    # type: ignore
from torch.utils.data import DataLoader

from neuralNetworkSmall import ConditionalUNet

from Diffusion import f, g, b

def teacher_integrate(model: nn.Module, x_batch: torch.Tensor, t_batch: torch.Tensor, delta_t: float, N: int) -> torch.Tensor:
    """
    Integrates the frozen teacher model backward from t to t - delta_t using N substeps.
    This implements the Euler-Maruyama Scheme also used for the SDE Sampler.
    """
    dt_sub = delta_t / N
    x_star = x_batch.clone()
    t_curr = t_batch.clone()
    
    with torch.no_grad():
        for _ in range(N):

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
            noise_injection = g_t * torch.sqrt(torch.tensor(dt_sub, device=device)) * torch.randn_like(x)
            
            # Continuous SDE reverse step formula
            x_star = x_star - drift_update + score_update + noise_injection
            t_curr -= dt_sub

    return x_star

def endpoint_distillation_step(
    teacher: nn.Module, 
    student: nn.Module, 
    optimizer: optim.Optimizer, 
    x: torch.Tensor, 
    t: torch.Tensor, 
    delta_t: float, 
    N: int
) -> float:

    # 1. Teacher integration to compute reference endpoint x*
    x_star = teacher_integrate(teacher, x, t, delta_t, N)
    
    # 2. Student takes one explicit Euler step over the full delta_t
    v_student = student(x, t)
    x_hat = x - v_student * delta_t
    
    # 3. Compute MSE Loss between student endpoint and teacher endpoint
    loss_fn = nn.MSELoss()
    loss = loss_fn(x_hat, x_star)
    
    # 4. Optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


# Example Usage
# Determine device and set up model and loss function accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher = ConditionalUNet(in_channels=1, out_channels=1).to(device)
teacher.load_state_dict(torch.load('./diffusion_model_mnist.pth', map_location=device))

student = ConditionalUNet(in_channels=1, out_channels=1).to(device)
student.load_state_dict(torch.load('./diffusion_model_mnist.pth', map_location=device))

transform = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),  # Scales to [0, 1]
    v2.ToTensor(),
    v2.Normalize(mean=[0.5], std=[0.5])  # Shifts to [-1, 1]
])

training_data = datasets.MNIST(    # type: ignore
    root="../data",
    train=True,
    download=True,
    transform=transform
)

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)    # type: ignore

optimizer = torch.optim.AdamW(student.parameters(), lr=2e-4, weight_decay=1e-4)

# Number of teacher substeps, i.e. distilling N teacher steps into 1 student step
N = 2

# Number of student steps, i.e. in the end we want to sample with 4096 steps
M = 4096
eps = 1e-3
linspace_of_endpoints = torch.linspace(1, eps, M)
delta_t = 1 / M

iterations = 10

teacher.eval()
student.train()
for _ in range(iterations):

    # sample a batch from the dataset
    x_batch = next(iter(train_dataloader))

    # sample a batch of endpoint time steps
    indices = torch.randint(0, M, (128,))
    t_batch = linspace_of_endpoints[indices]

    # noisify x_batch according to t_batch

    # integrate backwards in time using the teacher method and N uniform substeps
    x_target = teacher_integrate(model=teacher, x_batch=x_batch, t_batch=t_batch, delta_t=delta_t, N=N)

    # integrate backwards in time using the student method and 1 substep

    # compute the loss and update the weights

    loss = endpoint_distillation_step(teacher, student, optimizer, x_batch, t_batch, y_batch, delta_t, N)