import argparse

def assert_dependencies(args: argparse.Namespace):

    if args.where == 'cluster':
        assert args.data_dir is not None, 'If "where" is set to cluster a "data-dir" has to be provided'

    if args.what != 'full' and args.what != 'train':
        assert args.model is not None, 'If no teacher is trained beforehand the path to a trained model has to be given'

    if args.sampling_sampler == 'em':
        assert args.which == 'diffusion', 'euler maruyama is only allowed for the diffuision model'

    if args.distill_teacher_sampler == 'em':
        assert args.which == 'diffusion', 'euler maruyama is only allowed for the diffusion model'

    if args.what == 'eval':
        assert args.eval_model_folder_id is not None

    if args.sampling_mode == '8x8':
        assert args.sampling_num_samples == 64

