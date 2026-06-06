"""Wraps around all other functionalities to provide a unified entrypoint to the training, evaluating and sampling for the 4 generative models.
# ! No automatic sampling and evaluation of a distilled student, that has to be handled externally
"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Model on CIFAR10")
    
    # ! general setup
    parser.add_argument('--which', type=str, choices=['diffusion', 'kac', 'mmd', 'föllmer'], default='diffusion',
                        help='which model do you want to run')
    parser.add_argument('--where', type=str, choices=['local', 'cluster'], default='local',
                        help='where do you want to run the model, locally or on some hpc cluster. Cluster is also possible if you have local cuda support.\
                            Youll have to adjust the paths though.')
    parser.add_argument('--what', type=str, choices=[ 'full', 'train', 'sample', 'eval', 'distill', 'train+eval'], default='full',
                        help='lets you adjust what exactly you want to run if you only need a certain segment')
    

    # ! training arguments
    # ! 1000 epochs to allow enough time for both patience mechanisms to terminate. This is only an upper bound
    parser.add_argument('--training-epochs', type=int, default=1_000,
                        help='specifies the amount of epochs in training, and which model to use in sampling and eval')
    parser.add_argument('--training-batch-size', type=int, default=256,
                        help='only needed for training')
    parser.add_argument('--training-stage2-samples', type=int, default=5_000,
                        help='lets you set a different sample size on which the fid checkpoints are calculated')
    parser.add_argument('--training-stage1-patience', type=float, default=0.1,
                        help='lets you set a different percentage for loss patience. Defaults to 10perc of total epochs')
    parser.add_argument('--training-stage2-patience', type=float, default=0.01,
                        help='lets you set a different percentage for fid patience. Defaults to 1perc of total epochs')
    

    # ! sampling arguments
    parser.add_argument('--sampling-sampler', type=str, choices=['ee', 'rk2', 'rk45', 'ab2', 'em'], default='ee',
                        help='chose a method to sample with, em is only available for diffusion models.')
    parser.add_argument('--sampling-mode', type=str, choices=['8x8', 'set'], default='set',
                        help='8x8 generates a 8x8 grid of samples to showcase the result, set generates a full set useful for fid evaluation')
    
    parser.add_argument('--sampling-num-steps', type=int, default=8_192,
                        help='if sampler uses linspace, this specifies the amount of steps. I.e. for diff with SDE, kac with ee or rk2')
    parser.add_argument('--sampling-batch-size', type=int, default=512,
                        help='specifies how many samples are to be processed at the same time. I.e. the tensor shape.')

    # configuration of the adaptive solver RK45
    parser.add_argument('--sampling-rel-tol', type=float, default=1e-4,
                        help='used for the RK45 solver which is employed for diff with pfode and kac with rk45')
    parser.add_argument('--sampling-abs-tol', type=float, default=1e-4,
                        help='used for the RK45 solver which is employed for diff with pfode and kac with rk45')

    # ! dual use for sampling and evaluating
    parser.add_argument('--sampling-num-samples', type=int, default=50_000,
                        help='only needed if sampler_mode is set to "set", specifies how many samples are to be generated')


    # ! distillation arguments
    parser.add_argument('--distill-iterations', type=int, default=50_000,
                        help='sets the amount of iterations the student model should be trained for')
    parser.add_argument('--distill-teacher-sampler', type=str, default='ee', choices=['ee', 'rk2', 'ab2', 'rk45', 'em'],
                        help='provides the possibility to set a different teacher sampler than previously defined, if not sets defaults to ee')
    parser.add_argument('--distill-student-sampler', type=str, default='ee', choices=['ee'],
                        help='provides the possibility to set a different student sampler than ee, if not set defaults to explicit euler')
    parser.add_argument('--distill-num-student-steps', type=int, default=512,
                        help='specifies the amount of steps the student should do in order to sample, i.e. a 20-step student or a 10-step student.')
    parser.add_argument('--distill-num-teacher-substeps', type=int, default=16,
                        help='the amount of teacher steps the student is supposed to learn')


    # ! general arguments
    parser.add_argument('--model', type=str,
                        help='path to model, relative or absolute, needed if "what" is set to "sample" or "eval" or "distill"')


    # ! configuration arguments
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='specifies the learning rate of the training process')
    parser.add_argument('--dataset', type=str, choices=['cifar10'], default='cifar10',
                        help='which dataset you want to train on options include [cifar10]')
    parser.add_argument('--T', type=float, default=1.0,
                        help='specifies the time horizon T')
    # * 1e-5 as specified by "Song et al 2021 - Score based generative modelling through sdes" and as referenced by "Duong Chemseddine 2025 - Telegraphers Generative Model via Kac Flows"
    parser.add_argument('--time-truncation', type=float, default=1e-5,
                        help='lets you set a cutoff time for the model, defaults to 1e-5, used for diffusion training and sampling, mmd sampling')
    

    # of kac
    # * a = 900, c = 10, g(t)=t**2 as specified by "Duong et al 2025 - Telegraphers Generetive Model via Kac Flows"
    parser.add_argument('--kac-a', type=float, default=900,
                        help='specifies the damping coefficient a of the kac process')
    parser.add_argument('--kac-c', type=float, default=10,
                        help='specifies the wave front speed c of the kac process')
    parser.add_argument('--kac-f', type=str, default='opt1', choices=['opt1'],
                        help='lets you choose different data schedules, opt1 is "1-t"')
    parser.add_argument('--kac-g', type=str, default='opt2', choices=['opt1', 'opt2'],
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

    # set some standard values
    if args.sampling_mode == '8x8':
        args.sampling_num_samples = 64

    if args.which == 'kac':
        args.time_truncation = 0

    args.lr = args.lr * (args.training_batch_size / 128)


    print(f'\nData directory:  {args.data_dir}')


    from Cluster.utils.dataHandling import DataProvider
    data = DataProvider(args=args)


    # Determine device and set up model and loss function accordingly
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    print(f'\nDetermined device:  {device}')


    # Set up the model path
    path_to_model = f"{args.where}_{args.which}_epochs{args.training_epochs}_model.pth"
    if args.where == 'cluster':
        path_to_model = f"/work/zastrau/{path_to_model}"

    if args.what != 'full' and args.what != 'train':    # then a path to a model has to be given through the arguments
        path_to_model = args.model
    print(f'\nDetermined model path:  {path_to_model}')


    # Set up the student model path
    path_to_distilled_student = f"{args.where}_{args.which}_epochs{args.training_epochs}_model_student.pth"
    if args.where == 'cluster':
        path_to_distilled_student = f"/work/zastrau/{path_to_distilled_student}"
    print(f'\nDetermined student model path:  {path_to_distilled_student}')


    # Set up model image path
    base_name = f"{args.which}_epochs{args.training_epochs}_sampler{args.sampling_sampler}"
    if args.sampling_sampler in ['ee', 'rk2', 'em', 'ab2']:    # then fixed step size
        base_name = f'{base_name}_steps{args.sampling_num_steps}'
    else:
        base_name = f'{base_name}_rk45'

    # Mode specific path extension
    if args.sampling_mode == '8x8':
        base_name = f'{base_name}_8x8.png'
    else:    # args.sampling_mode == 'set'
        base_name = f'{base_name}_set'
        

    # Set up student image path
    student_base_name = f"{args.which}_iterations{args.distill_iterations}_sampler{args.sampling_sampler}"
    if args.sampling_sampler in ['ee', 'rk2', 'em', 'ab2']:    # then fixed step size
        student_base_name = f'{student_base_name}_steps{args.sampling_num_steps}'
    else:
        student_base_name = f'{student_base_name}_rk45'

    # Mode specific path extension
    if args.sampling_mode == '8x8':
        student_base_name = f'{student_base_name}_8x8.png'
    else:    # args.sampling_mode == 'set'
        student_base_name = f'{student_base_name}_set'


    # Location specific path start
    save_path = f"./{base_name}"
    student_save_path = f'./{student_base_name}'
    if args.where == 'cluster':
        if args.sampling_mode == '8x8':
            save_path = f"/homes/math/zastrau/NeuralNetworkSamples/{base_name}"
            student_save_path = f'/homes/math/zastrau/NeuralNetworkSamples/{student_base_name}'
        else:    # args.sampling_mode == 'set'
            save_path = f"/work/zastrau/samples/{base_name}"
            student_save_path = f'/work/zastrau/samples/{student_base_name}'
    print(f'\nDetermined teacher image path: {save_path}')
    print(f'\nDetermined student image path: {student_save_path}')


    # Set up model based on location
    if args.where == 'cluster':
        from Cluster.utils.modelGetter import model_getter
        model = model_getter(args=args).to(device)

        size = 'large'
    else:    # args.where == 'local'
        from Cluster.networks.neuralNetworkSmall import ConditionalUNet
        model = ConditionalUNet(in_channels=data.data_dims.channels, out_channels=data.data_dims.channels).to(device)

    if args.what in ['eval', 'sample']:
        model.load_state_dict(torch.load(path_to_model, map_location=device))


    # compile the model to fuse and optimize the UNet graph for the GPU
    if args.where == 'cluster':
        model = torch.compile(model)


    # Set up sampler if needed
    sampler = None
    if args.which == 'kac':
        from Cluster.utils.sample_kac import TorchKacConstantSampler
        sampler = TorchKacConstantSampler(a=args.kac_a, c=args.kac_c, T=args.T, M=50000, K=4096)
    

    from Cluster.utils.reversals import Reversal
    reversal_fns = Reversal(args=args)


    from Cluster.utils.lossFunctions import LossFns
    loss_fn = LossFns(args=args, sampler=sampler)


    from Cluster.utils.noisifier import Noisify
    noisify_fns = Noisify(args=args)


    # Train the model
    if args.what in ['full', 'train', 'train+eval']:
        print('----------------------------------------------------------------------------------------------------')
        print(f'\nStarting the training')

        from Cluster.training import training_wrapper
        training_wrapper(args=args, loss_fn=loss_fn, reversal_fns=reversal_fns, model=model, data=data, save_path=path_to_model)


    # Sample from the model
    if args.what in ['full', 'sample', 'train+eval']:
        print('----------------------------------------------------------------------------------------------------')
        print(f'\nStarting the sampling for {args.which} with {args.sampling_sampler}, sampling {args.sampling_num_samples} samples.')

        from Cluster.sampling import sample_wrapper
        sample_wrapper(args=args, model=model, data=data, sampler=sampler, reversal_fns=reversal_fns, save_path=save_path)


    # Evaluate the model using FID
    if args.what in ['full', 'eval', 'train+eval']:
        print('----------------------------------------------------------------------------------------------------')
        print(f'\nEvaluating the model {path_to_model}')

        from Cluster.eval import eval_wrapper
        eval_wrapper(args=args, data=data, img_path=save_path)


        print(f'\nFinished evaluation.')

    if args.what in ['full', 'distill']:
        print('----------------------------------------------------------------------------------------------------')
        print(f'\nDistilling the teacher model {path_to_model} into a {args.distill_num_student_steps} step student.')

        # TODO Need to also implement distillation for all other processes, Schrödinger
        from Cluster.distillation import distillation_wrapper
        student_model = distillation_wrapper(
            args=args, save_path=path_to_distilled_student, model_path=path_to_model,
            reversal_fns=reversal_fns, noisify_fns=noisify_fns
        )

        print(f'\nFinished Distillation.')