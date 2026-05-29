"""Wraps around all other functionalities to provide a unified entrypoint to the training, evaluating and sampling for the 4 generative models."""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Model on CIFAR10")
    
    # ! general setup
    parser.add_argument('--which', type=str, choices=['diffusion', 'kac', 'mmd', 'föllmer'], default='diffusion',
                        help='which model do you want to run')
    parser.add_argument('--where', type=str, choices=['local', 'cluster'], default='local',
                        help='where do you want to run the model, locally or on some hpc cluster. Cluster is also possible if you have local cuda support.\
                            Youll have to adjust the paths though.')
    parser.add_argument('--what', type=str, choices=['train', 'sample', 'eval', 'full', 'sampleeval', 'distill'], default='full',
                        help='lets you adjust what exactly you want to run if you only need a certain segment')
    

    # ! training arguments
    parser.add_argument('--epochs', type=int, default=200,
                        help='specifies the amount of epochs in training, and which model to use in sampling and eval')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='only needed for training')
    

    # ! sampling arguments
    parser.add_argument('--sampler-diff', type=str, choices=['sde', 'pfode'], default='sde',
                        help='only required if the "which" flag is set to "diffusion" defaults to euler-maruyama scheme of the reverse time SDE')
    parser.add_argument('--sampler-kac', type=str, choices=['ee', 'rk2', 'rk45'], default='rk45',
                        help='only required if the "which" flag is set to "kac" defaults to flowODE with rk45')    # TODO: might delete this since I only want to be using RK45
    parser.add_argument('--sampler-mode', type=str, choices=['8x8', 'set'], default='set',
                        help='8x8 generates a 8x8 grid of samples to showcase the result, set generates a full set useful for fid evaluation')
    
    parser.add_argument('--num-teacher-steps', type=int, default=8192,
                        help='if sampler uses linspace, this specifies the amount of steps. I.e. for diff with SDE, kac with ee or rk2')
    
    # configuration of the adaptive solver RK45
    parser.add_argument('--rel-tol', type=float, default=1e-4,
                        help='used for the RK45 solver which is employed for diff with pfode and kac with rk45')
    parser.add_argument('--abs-tol', type=float, default=1e-4,
                        help='used for the RK45 solver which is employed for diff with pfode and kac with rk45')


    # ! dual use for sampling and evaluating
    parser.add_argument('--num-samples', type=int, default=50_000,
                        help='only needed if sampler_mode is set to "set", specifies how many samples are to be generated')


    # ! distillation arguments
    parser.add_argument('--iterations', type=int, default=100,
                        help='sets the amount of iterations the student model should be trained for')
    parser.add_argument('--num-student-steps', type=int, default=1024,
                        help='specifies the amount of steps the student should do in order to sample, i.e. a 20-step student or a 10-step student.')
    parser.add_argument('--num-teacher-substeps', type=int, default=10,
                        help='the amount of teacher steps the student is supposed to learn')
    parser.add_argument('--model', type=str,
                        help='specifies the path to the model which is supposed to be distilled. Absolute or relative to the execution location')


    # ! configuration arguments
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='specifies the learning rate of the training process')
    parser.add_argument('--dataset', type=str, choices=['cifar10'], default='cifar10',
                        help='which dataset you want to train on options include [cifar10]')
    
    # of kac
    parser.add_argument('--a', type=float, default=9.0,
                        help='specifies the damping coefficient a of the kac process')
    parser.add_argument('--c', type=float, default=3.0,
                        help='specifies the wave front speed c of the kac process')
    parser.add_argument('--T', type=float, default=1.0,
                        help='specifies the time horizon T of the kac process')
    
    # cluster
    parser.add_argument('--data-dir', type=str,
                        help='the local directory of the node the job runs on in the cluster')

    args = parser.parse_args()


    # Checking if argument dependencies are fulfilled
    from Cluster.argumentDependencyChecker import assert_dependencies
    assert_dependencies(args=args)


    print(f'\nData directory:  {args.data_dir}\n')


    from Cluster.utils.dataHandling import DataProvider
    data = DataProvider(args=args)


    # Determine device and set up model and loss function accordingly
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'\nDetermined device:  {device}\n')


    # Set up the model path
    path_to_model = f"{args.where}_{args.which}_epochs{args.epochs}_model.pth"
    if args.where == 'cluster':
        path_to_model = f"/work/zastrau/{path_to_model}"
    print(f'\nDetermined model path:  {path_to_model}\n')


    # Set up the student model path
    path_to_distilled_student = f"{args.where}_{args.which}_{args.epochs}_model_student.pth"
    if args.where == 'cluster':
        path_to_distilled_student = f"/work/zastrau/{path_to_distilled_student}"
    print(f'\nDetermined student model path:  {path_to_distilled_student}\n')


    # Set up image path
    base_name = f"{args.which}_epochs{args.epochs}"
    # Process specific path extension
    if args.which == 'diffusion':
        base_name = f"{base_name}_sampler{args.sampler_diff}"
        if args.sampler_diff == 'sde':
            base_name = f"{base_name}_steps{args.num_teacher_steps}"

    else:    # args.which == 'kac'
        base_name = f"{base_name}_sampler{args.sampler_kac}"
        if args.sampler_kac in ['ee', 'rk2']:    # then fixed step size
            base_name = f'{base_name}_steps{args.num_steps}'

    # Mode specific path extension
    if args.sampler_mode == '8x8':
        base_name = f'{base_name}_8x8.png'
    else:    # args.sampler_mode == 'set'
        base_name = f'{base_name}_set'
        
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
        from Cluster.neuralNetworkOpenAI import UNetModel
        model = UNetModel(
            image_size=data.data_dims.size, in_channels=data.data_dims.channels,
            out_channels=data.data_dims.channels,
            model_channels=128, num_res_blocks=2,
            attention_resolutions=[16], num_heads=4,
            num_head_channels=64, channel_mult=(1, 2, 2, 2)
        ).to(device)

        size = 'large'
    else:    # args.where == 'local'
        from Cluster.neuralNetworkSmall import ConditionalUNet
        model = ConditionalUNet(in_channels=data.data_dims.channels, out_channels=data.data_dims.channels).to(device)

        size = 'small'

    if args.what in ['eval', 'sample']:
        model.load_state_dict(torch.load(path_to_model, map_location=device))
    print(f'\nInstantiated the {size} model\n')


    # compile the model to fuse and optimize the UNet graph for the GPU
    if args.where == 'cluster':
        model = torch.compile(model)


    # Set up sampler if needed
    sampler = None
    if args.which == 'kac':
        from Cluster.utils.sample_kac import TorchKacConstantSampler
        
        sampler = TorchKacConstantSampler(a=args.a, c=args.c, T=args.T, M=50000, K=4096)
        print('\nInstantiated the kac sampler\n')
    

    # Train the model
    if args.what in ['full', 'train']:
        print('----------------------------------------------------------------------------------------------------')
        print(f'\nStarting the training\n')

        from Cluster.training import training_wrapper
        from Cluster.utils.lossFunctions import LossFns

        print(f'\nInstantiating the loss function\n')
        loss_fn: object = LossFns(args=args, sampler=sampler)
        training_wrapper(args=args, loss_fn=loss_fn, model=model, data=data, save_path=path_to_model)


    # Sample from the model
    if args.what in ['full', 'sample', 'sampleeval']:
        print('----------------------------------------------------------------------------------------------------')
        print(f'\nStarting the sampling for {args.which} with {args.sampler_diff if args.which == 'diffusion' else args.sampler_kac}\n')

        if args.which == 'diffusion':
            if args.sampler_diff == 'sde':
                from Cluster.diffusionInferenceSDE import sample_wrapper
            else:    # args.sampler == 'pfode'
                from Cluster.diffusionInferencePFODE import sample_wrapper
        else:    # args.which == 'kac'
            from Cluster.kacInferenceODE_Jannis import sample_wrapper

        sample_wrapper(args=args, model=model, data=data, sampler=sampler, save_path=save_path)


    # Evaluate the model using FID
    if args.what in ['full', 'eval', 'sampleeval']:
        print('----------------------------------------------------------------------------------------------------')
        print(f'\nEvaluating the model {path_to_model}\n')

        from Cluster.eval import eval_wrapper
        eval_wrapper(args=args, data=data, img_path=save_path)


    if args.what in ['full', 'distill']:
        print('----------------------------------------------------------------------------------------------------')
        print(f'\nDistilling the {args.num_teacher_steps}step teacher model {path_to_model} into a {args.num_student_steps}step student')

        # TODO Need to also implement distillation for all other processes, MMD and Schrödinger and Kac
        from Cluster.distillation import distillation_wrapper
        distillation_wrapper(args=args, save_path=path_to_distilled_student, model_path=path_to_model)