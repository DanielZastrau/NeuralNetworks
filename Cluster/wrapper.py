import argparse

import torch

from lossFunctions import LossFns
from training import training_wrapper

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Model on CIFAR10")
    
    parser.add_argument('--which', type=str, choices=['diffusion', 'kac', 'mmd', 'föllmer'], required=True)
    parser.add_argument('--where', type=str, required=True, choices=['local', 'cluster'])
    parser.add_argument('--what', type=str, default='train', choices=['train', 'eval', 'sample', 'full'])
    parser.add_argument('--sampler_diff', type=str, choices=['sde', 'pfode'], default='sde')
    parser.add_argument('--sampler_kac', type=str, choices=['ee', 'rk2', 'rk45', 'self'], default='rk45')    # TODO: might delete this since I only want to be using RK45
    parser.add_argument('--epochs', type=int, default=200, help='only needed for training')
    parser.add_argument('--batch_size', type=int, default=512, help='only needed for training')
    parser.add_argument('--num_steps', type=int, default=8192, help='if sampler uses linspace, this specifies the amount of steps. I.e. for diffSDE, kacODE with ee or rk2')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--a', type=float, default=9.0)
    parser.add_argument('--c', type=float, default=3.0)
    parser.add_argument('--T', type=float, default=1.0)
    parser.add_argument('--rel_tol', type=float, default=1e-3, help='used for the RK45 solver which is employed for diffSDE and kacODE')
    parser.add_argument('--abs_tol', type=float, default=1e-3)
    
    args = parser.parse_args()

    # Determine device and set up model and loss function accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'\nDetermined device:  {device}\n')

    # Set up the model path
    path_to_model = f"{args.where}_{args.which}_{args.epochs}_model.pth"
    if args.where == 'cluster':
        path_to_model = f"/work/zastrau/{path_to_model}.pth"
    print(f'\nDetermined path:  {path_to_model}\n')


    # Set up model based on location
    if args.where == 'cluster':
        from neuralNetworkOpenAI import UNetModel
        model = UNetModel(
            image_size=32, in_channels=3,
            out_channels=3, model_channels=64,
            num_res_blocks=5, attention_resolutions=[16, 32],
            num_heads=4
        ).to(device)
    else:    # args.where == 'local'
        from neuralNetworkSmall import ConditionalUNet
        model = ConditionalUNet(in_channels=3, out_channels=3).to(device)

    if args.what in ['eval', 'sample']:
        model.load_state_dict(torch.load(path_to_model, map_location=device))
    print('\nInstantiated the model\n')


    # Set up sampler if needed
    sampler = None
    if args.which == 'kac':
        from utils.sample_kac import TorchKacConstantSampler
        
        sampler = TorchKacConstantSampler(a=args.a, c=args.c, T=args.T, M=50000, K=4096)
        print('\nInstantiated the kac sampler\n')
    
    # Train the model
    if args.what in ['full', 'train']:

        print(f'\nInstantiating the loss function\n')
        loss_fn: object = LossFns(args=args, sampler=sampler)

        print(f'\nStarting the training\n')
        training_wrapper(args=args, loss_fn=loss_fn, model=model, save_path=path_to_model)


    # Evaluate the model using FID
    if args.what in ['full', 'eval']:
        pass


    # Sample from the model
    if args.what in ['full', 'sample']:

        if args.which == 'diffusion':
            if args.sampler_diff == 'sde':
                from DiffusionInferenceSDE import sample_wrapper
            else:    # args.sampler == 'pfode'
                from DiffusionInferencePFODE import sample_wrapper
        else:    # args.which == 'kac'
            if args.sampler_kac == 'self':
                from kacInferenceODE_self import sample_wrapper
            else:    # args.sampler_kac in ['ee', 'rk2', 'rk45']
                from kacInferenceODE_Jannis import sample_wrapper

        print(f'\nStarting the sampling for {args.which} with {args.sampler_diff if args.which == 'diffusion' else args.sampler_kac}\n')
        sample_wrapper(args=args, model=model, sampler=sampler)
