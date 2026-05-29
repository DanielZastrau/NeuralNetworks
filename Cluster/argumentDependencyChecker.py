import argparse

def assert_dependencies(args: argparse.Namespace):

    if args.where == 'cluster':
        assert args.data_dir is not None, 'If "where" is set to cluster a "data-dir" has to be provided'

    if args.what != 'full':
        assert args.model is not None, 'If no teacher is trained beforehand the path to a trained model has to be given'