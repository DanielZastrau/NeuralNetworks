"""Wraps around all other functionalities to provide a unified entrypoint to the training, evaluating and sampling for the 4 generative models.
# ! No automatic sampling and evaluation of a distilled student, that has to be handled externally
"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Model on CIFAR10")
    
    # ! general setup
    parser.add_argument('--which', type=str, choices=['diffusion', 'kac', 'mmd', 'schrödinger'], required=True)
    parser.add_argument('--where', type=str, choices=['local', 'cluster'], default='local', help='Currently only works for very specific paths.')
    parser.add_argument('--what', type=str, choices=['train', 'sample', 'eval', 'distill'], required=True)
    

    # ! training arguments
    # * We adopt the training protocoll of "2025 - Duong et al - Telegraphers" which just trains for 400k iterations
    # * batch sizes of 128 seem to be the standard, see: "2025 - Duong et al - Telegraphers", "2025 - Han et al DistillKac"
    parser.add_argument('--training-iterations', type=int, default=400_000)
    parser.add_argument('--training-logging-period', type=int, default=1_000,
                        help='lets you set the regular interval where the process sends you a lifesign')
    parser.add_argument('--training-batch-size', type=int, default=128)
    
    parser.add_argument('--training-evaluation-period-loss', type=int, default=2_000)

    parser.add_argument('--training-evaluation-period-grid', type=int, default=10_000)
    parser.add_argument('--training-evaluation-period-grid-num-steps', type=int, default=1_024)
    parser.add_argument('--training-evaluation-period-grid-sampler', type=str, default='ee', choices=['ee', 'rk2', 'em'], help='EM only for diffusion.')

    parser.add_argument('--training-evaluation-period-fid', type=int, default=50_000)
    parser.add_argument('--training-evaluation-period-fid-num-samples', type=int, default=2_000)
    parser.add_argument('--training-evaluation-period-fid-num-steps', type=int, default=1_024)
    parser.add_argument('--training-evaluation-period-fid-sampler', type=str, default='ee', choices=['ee', 'rk2', 'em'], help='EM only for diffusion.')

    parser.add_argument('--training-optimizer', type=str, default="adam", choices=['adam', 'adamW'])
    parser.add_argument('--training-optimizer-weight-decay', type=float, default=0.01, help='only used for adamW')
    parser.add_argument('--training-optimizer-lr', type=float, default=2e-4)

    parser.add_argument('--training-scheduler', type=str, default='cosine', choices=['cosine', 'constant'])

    parser.add_argument('--training-verbosity', default='normal', choices=['silent', 'normal', 'verbose'])


    # ! sampling arguments
    # * 2025 - Duong et al - Telegraphers Generative Model via Kac Flows, they seem to sample kac with 100 steps
    parser.add_argument('--sampling-sampler', type=str, choices=['ee', 'rk2', 'em'], default='ee', help='EM only for diffusion.')
    parser.add_argument('--sampling-mode', type=str, choices=['8x8', 'set'], default='set',
                        help='8x8 generates a 8x8 grid of samples to showcase the result, set generates a full set useful for fid evaluation')
    
    parser.add_argument('--sampling-num-steps', type=int, default=1_024,
                        help='if sampler uses linspace, this specifies the amount of steps. I.e. for diff with SDE, kac with ee or rk2')
    parser.add_argument('--sampling-batch-size', type=int, default=512,
                        help='specifies how many samples are to be processed at the same time. I.e. the tensor shape.')
    parser.add_argument('--sampling-num-samples', type=int, default=50_000,
                        help='only needed if sampler_mode is set to "set", specifies how many samples are to be generated')

    # configuration of the adaptive solver RK45
    parser.add_argument('--sampling-rel-tol', type=float, default=1e-4,
                        help='used for the RK45 solver which is employed for diff with pfode and kac with rk45')
    parser.add_argument('--sampling-abs-tol', type=float, default=1e-4,
                        help='used for the RK45 solver which is employed for diff with pfode and kac with rk45')
    
    parser.add_argument('--sampling-logging-period', type=int, default=1_000,
                        help='lets you set the regular interval where the process sends you a lifesign')


    # ! eval arguments
    parser.add_argument('--eval-sampler', type=str, default='ee', choices=['ee', 'rk2', 'em'], help='em only for diffusion')
    parser.add_argument('--eval-num-steps', type=int, default=1_024)
    parser.add_argument('--eval-num-samples', type=int, default=50_000)
    parser.add_argument('--eval-model-folder-id', type=int)
    parser.add_argument('--eval-model-name', type=str)

    # ! distillation arguments
    parser.add_argument('--distill-iterations', type=int, default=400_000)
    parser.add_argument('--distill-teacher-sampler', type=str, default='ee', choices=['ee', 'rk2', 'ab2', 'rk45', 'em'])
    parser.add_argument('--distill-student-sampler', type=str, default='ee', choices=['ee'])
    parser.add_argument('--distill-num-student-steps', type=int, help='The amount of steps the student should take in total.')
    parser.add_argument('--distill-num-teacher-substeps', type=int, help='The amount of teacher substeps the student is supposed to learn')
    parser.add_argument('--distill-lr', type=float, default=2e-4)
    parser.add_argument('--distill-weight-decay', type=float, default=0.01)
    parser.add_argument('--distill-model-folder-id', type=int)
    parser.add_argument('--distill-model-name', type=str)


    # ! general arguments
    parser.add_argument('--model', type=str, default='',
                        help='path to model, relative or absolute, needed if "what" is set to "sample" or "eval" or "distill", optional for "what" = "train"')


    # ! configuration arguments
    parser.add_argument('--dataset', type=str, choices=['cifar10'], default='cifar10',
                        help='which dataset you want to train on options include [cifar10]')
    parser.add_argument('--T', type=float, default=1.0,
                        help='specifies the time horizon T')
    # * 1e-5 as specified by "Song et al 2021 - Score based generative modelling through sdes" and as referenced by "Duong Chemseddine 2025 - Telegraphers Generative Model via Kac Flows"
    parser.add_argument('--time-truncation', type=float, default=1e-8,
                        help='lets you set a cutoff time for the model, defaults to 1e-5, used for diffusion training and sampling, mmd sampling')
    

    # of kac
    # * a = 25, c = 2, g(t)=t**2 as specified by "2025 - Duong et al - Telegraphers Generetive Model via Kac Flows" as the best parameters
    # * these are also used by "2025 - Han et al - DistillKac"
    parser.add_argument('--kac-a', type=float, default=25,
                        help='specifies the damping coefficient a of the kac process')
    parser.add_argument('--kac-c', type=float, default=2,
                        help='specifies the wave front speed c of the kac process')
    parser.add_argument('--kac-f', type=str, default='opt1', choices=['opt1'],
                        help='lets you choose different data schedules, opt1 is "1-t"')
    parser.add_argument('--kac-g', type=str, default='opt1', choices=['opt1', 'opt2'],
                        help='lets you choose different noise schedules, opt1 is "t",  opt2 is "t^2"')


    # of mmd
    parser.add_argument('--mmd-b', type=int, default=3,
                        help='sets the uniform distribution towards which the process moves')
    parser.add_argument('--mmd-f', type=str, default='opt1',
                        help='lets you select another data schedule f, doesnt do anything right now, only there as a placeholder for the future')
    parser.add_argument('--mmd-g', type=str, default='opt1',
                        help='lets you select another noise schedule f, doesnt do anything right now, only there as a placeholder for the future')

    # cluster
    parser.add_argument('--data-dir', type=str,
                        help='the local directory of the node the job runs on in the cluster')

    # ! Dev
    parser.add_argument('--proof-of-concept', action='store_true',
                        help='exists so that I can cut training to exactly one iteration, in case I just want to see the full framework run start to end without apparent errors')

    args = parser.parse_args()


    # Checking if argument dependencies are fulfilled
    from Cluster.utils.argumentDependencyChecker import assert_dependencies
    assert_dependencies(args=args)

    from Cluster.utils.argumentStandards import set_standards
    args = set_standards(args=args)

    print('-'*100)
    print(args)
    print('-'*100)

    import torch
    torch.set_float32_matmul_precision('highest')

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'\nDetermined device:  {device}')


    # ! if args.what == sample, then previously a model had to be trained and thus the corresponding folders already exist
    # ! if args.what == distill, then the same
    # ! if args.what == eval, then no folders are needed
    if args.what in ['train']:
        # Get run Id
        import random
        idx = random.randint(0, 10_000)
        print(f'ID: {idx}')
        
        # Determine paths
        base = ''
        if args.where == 'cluster':
            base = f'/work/zastrau/{idx}'
        else:
            base = f'./{idx}'
            
        import os
        # path for periodic grid generation in training
        grid_path = os.path.join(base, 'grid')

        # path for models
        model_path = os.path.join(base, 'models')

        # path for the sampling module
        images_path = os.path.join(base, 'samples')

        # Ensure that the path exists
        if not os.path.exists(base):
            os.mkdir(base)
        if not os.path.exists(grid_path):
            os.mkdir(grid_path)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        if not os.path.exists(images_path):
            os.mkdir(images_path)


    from Cluster.utils.dataHandling import DataProvider
    data = DataProvider(args=args)

    # Set up model based on location
    if args.where == 'cluster':
        from Cluster.utils.modelGetter import model_getter
        model = model_getter(args=args).to(device)
        model.convert_to_fp32()

    else:    # args.where == 'local'
        from Cluster.networks.neuralNetworkSmall import ConditionalUNet
        model = ConditionalUNet(in_channels=data.data_dims.channels, out_channels=data.data_dims.channels).to(device)

    print(f'\nInstantiated the model.')



    # Train the model
    if args.what in ['train']:
        print('----------------------------------------------------------------------------------------------------')
        print(f'\nStarting the training')

        # Set up sampler if needed
        sampler = None
        if args.which == 'kac':
            from Cluster.utils.sample_kac import TorchKacConstantSampler
            sampler = TorchKacConstantSampler(a=args.kac_a, c=args.kac_c, T=args.T, M=50000, K=4096)

        from Cluster.utils.lossFunctions import LossFns
        loss_fn = LossFns(args=args, sampler=sampler, data=data)

        # compile the model to fuse and optimize the UNet graph for the GPU
        if args.where == 'cluster':
            model = torch.compile(model)

        from Cluster.training import training_wrapper
        training_wrapper(args=args, loss_fn=loss_fn, model=model, data=data, grid_path=grid_path, model_path=model_path)



    # Sample from the model
    if args.what in ['sample']:
        print('----------------------------------------------------------------------------------------------------')
        print(f'\nStarting the sampling for {args.which} with {args.sampling_sampler}, sampling {args.sampling_num_samples} samples.')

        # Set up sampler if needed
        sampler = None
        if args.which == 'kac':
            from Cluster.utils.sample_kac import TorchKacConstantSampler
            sampler = TorchKacConstantSampler(a=args.kac_a, c=args.kac_c, T=args.T, M=50000, K=4096)

        from Cluster.utils.reversals import Reversal
        reversal_fns = Reversal(args=args, which='sample')

        from Cluster.sampling import sample_wrapper
        sample_wrapper(args=args, model=model, data=data, sampler=sampler, reversal_fns=reversal_fns, save_path=images_path)

        print(f'\nFinished the sampling for {args.which} with {args.sampling_sampler}, sampling {args.sampling_num_samples} samples. And saved to {images_path}.')



    # Evaluate the model using FID
    if args.what in ['eval']:
        print('----------------------------------------------------------------------------------------------------')
        print(f'\nEvaluating the {args.eval_model_name} from the folder  {args.eval_model_folder_id}.')

        # Load the checkpoint file
        import os
        path = f'/work/zastrau/{args.eval_model_folder_id}/models/{args.eval_model_name}.pth'
        checkpoint = torch.load(path, map_location=device)
        print(f'Model path:  {path}.')

        # Check if it's a state_dict (a dictionary) or the full model object
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
            print('\nLoaded the state dict.')
        else:    # ! for wrongly saved older checkpoints
            model = checkpoint
            print('\nLoaded the full model object directly.')

        # compile the model to fuse and optimize the UNet graph for the GPU
        if args.where == 'cluster':
            model = torch.compile(model)

        from Cluster.utils.reversals import Reversal
        reversal_fns = Reversal(args=args, which='eval')

        # Set up sampler if needed
        sampler = None
        if args.which == 'kac':
            from Cluster.utils.sample_kac import TorchKacConstantSampler
            sampler = TorchKacConstantSampler(a=args.kac_a, c=args.kac_c, T=args.T, M=50000, K=4096)

        from Cluster.eval import eval_wrapper
        eval_wrapper(args=args, data=data, model=model, sampler=sampler, reversal_fns=reversal_fns)


        print(f'\nFinished evaluation.')


    if args.what in ['distill']:
        print('----------------------------------------------------------------------------------------------------')
        print(f'\nDistilling {args.distill_model_name} from folder {args.distill_model_folder_id} into a {args.distill_num_student_steps} step student\
                with {args.distill_num_teacher_substeps} teacher substeps.')

        # Load the checkpoint file
        import os
        path = f'/work/zastrau/{args.distill_model_folder_id}/models/{args.distill_model_name}.pth'
        checkpoint = torch.load(path, map_location=device)
        print(f'Model path:  {path}.')

        # Check if it's a state_dict (a dictionary) or the full model object
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
            print('\nLoaded the state dict.')
        else:    # ! for wrongly saved older checkpoints
            model = checkpoint
            print('\nLoaded the full model object directly.')

        import copy
        teacher = model
        student = copy.deepcopy(teacher)

        # compile the model to fuse and optimize the UNet graph for the GPU
        if args.where == 'cluster':
            teacher = torch.compile(model)
            student = torch.compile(student)

        from Cluster.distillation import distillation_wrapper
        student_model = distillation_wrapper(args=args, teacher=teacher, student=student, path=f'/work/zastrau/{args.distill_model_folder_id}/models/')

        print(f'\nFinished Distillation.')