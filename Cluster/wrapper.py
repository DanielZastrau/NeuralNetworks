import argparse
# This print exists because the cluster behaved weirdly
print('wrapper is started', flush=True)

import torch

if __name__ == "__main__":
    # This print exists because the cluster behaved weirdly
    print('entered the main block', flush=True)

    parser = argparse.ArgumentParser(description="Train Model on CIFAR10")
    
    parser.add_argument('--which', type=str, choices=['diffusion', 'kac', 'mmd', 'föllmer'], default='diffusion',
                        help='which model do you want to run')
    parser.add_argument('--where', type=str, choices=['local', 'cluster'], default='local',
                        help='where do you want to run the model, locally or on some hpc cluster. Cluster is also possible if you have local cuda support.\
                            Youll have to adjust the paths though.')
    parser.add_argument('--what', type=str, choices=['train', 'sample', 'eval', 'full', 'sampleeval'], default='full',
                        help='lets you adjust what exactly you want to run if you only need a certain segment')
    parser.add_argument('--sampler_diff', type=str, choices=['sde', 'pfode'], default='sde',
                        help='only required if the "which" flag is set to "diffusion" defaults to euler-maruyama scheme of the reverse time SDE')
    parser.add_argument('--sampler_kac', type=str, choices=['ee', 'rk2', 'rk45'], default='rk45',
                        help='only required if the "which" flag is set to "kac" defaults to flowODE with rk45')    # TODO: might delete this since I only want to be using RK45
    parser.add_argument('--sampler_mode', type=str, choices=['8x8', 'set'], default='set',
                        help='8x8 generates a 8x8 grid of samples to showcase the result, set generates a full set useful for fid evaluation')
    parser.add_argument('--epochs', type=int, default=200,
                        help='specifies the amount of epochs in training, and which model to use in sampling and eval')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='only needed for training')
    parser.add_argument('--num_steps', type=int, default=8192,
                        help='if sampler uses linspace, this specifies the amount of steps. I.e. for diff with SDE, kac with ee or rk2')
    parser.add_argument('--num_samples', type=int, default=128,
                        help='only needed if sampler_mode is set to "set", specifies how many samples are to be generated')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='specifies the learning rate of the training process')
    parser.add_argument('--a', type=float, default=9.0,
                        help='specifies the damping coefficient a of the kac process')
    parser.add_argument('--c', type=float, default=3.0,
                        help='specifies the wave front speed c of the kac process')
    parser.add_argument('--T', type=float, default=1.0,
                        help='specifies the time horizon T of the kac process')
    parser.add_argument('--rel_tol', type=float, default=1e-3,
                        help='used for the RK45 solver which is employed for diff with pfode and kac with rk45')
    parser.add_argument('--abs_tol', type=float, default=1e-3,
                        help='used for the RK45 solver which is employed for diff with pfode and kac with rk45')
    
    args = parser.parse_args()

    # Determine device and set up model and loss function accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'\nDetermined device:  {device}\n')


    # Set up the model path
    path_to_model = f"{args.where}_{args.which}_{args.epochs}_model.pth"
    if args.where == 'cluster':
        path_to_model = f"/work/zastrau/{path_to_model}.pth"
    print(f'\nDetermined model path:  {path_to_model}\n')


    # Set up image path
    base_name = f"{args.which}_epochs{args.epochs}_"
    # Process specific path extension
    if args.which == 'diffusion':
        base_name = f"{base_name}_sampler{args.sampler_diff}_"
        if args.sampler_diff == 'sde':
            base_name = f"{base_name}steps{args.num_steps}_"

    else:    # args.which == 'kac'
        base_name = f"{base_name}_sampler{args.sampler_kac}_"
        if args.sampler_kac in ['ee', 'rk2']:    # then fixed step size
            base_name = f'{base_name}steps{args.num_steps}'

    # Mode specific path extension
    if args.sampler_mode == '8x8':
        base_name = f'{base_name}8x8.png'
    else:    # args.sampler_mode == 'set'
        base_name = f'{base_name}set'
        
    # Location specific path start
    save_path = f"./{base_name}"
    if args.where == 'cluster':
        if args.sampler_mode == '8x8':
            save_path = f"/homes/math/zastrau/NeuralNetworkSamples/{base_name}"
        else:    # args.sampler_mode == 'set'
            save_path = f"/work/zastrau/samples/{base_name}"
    print(f'\nDetermined image path: {save_path}\n')


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
        from training import training_wrapper
        from lossFunctions import LossFns

        print(f'\nInstantiating the loss function\n')
        loss_fn: object = LossFns(args=args, sampler=sampler)

        print(f'\nStarting the training\n')
        training_wrapper(args=args, loss_fn=loss_fn, model=model, save_path=path_to_model)


    # Sample from the model
    if args.what in ['full', 'sample', 'sampleeval']:

        if args.which == 'diffusion':
            if args.sampler_diff == 'sde':
                from diffusionInferenceSDE import sample_wrapper
            else:    # args.sampler == 'pfode'
                from diffusionInferencePFODE import sample_wrapper
        else:    # args.which == 'kac'
            from kacInferenceODE_Jannis import sample_wrapper

        print(f'\nStarting the sampling for {args.which} with {args.sampler_diff if args.which == 'diffusion' else args.sampler_kac}\n')
        sample_wrapper(args=args, model=model, sampler=sampler, save_path=save_path)


    # Evaluate the model using FID
    if args.what in ['full', 'eval', 'sampleeval']:
        from eval import eval_wrapper

        eval_wrapper(args=args, img_path=save_path)
