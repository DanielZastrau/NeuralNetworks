import argparse

def set_standards(args: argparse.Namespace):

    if args.which == 'kac':
        # * as was done in "2025 - Duong et al - Telegraphers"
        args.time_truncation = 0

    args.training_optimizer_lr = args.training_optimizer_lr * (args.training_batch_size / 128)

    return args