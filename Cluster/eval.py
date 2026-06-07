import argparse

import torch
import torch_fidelity

from Cluster.sampling import sample
from Cluster.utils.dataHandling import DataProvider
from Cluster.utils.sample_kac import TorchKacConstantSampler
from Cluster.utils.reversals import Reversal
from Cluster.utils.uint8_utils import Uint8Dataset, to_uint8_rgb


def evaluate_fid(args: argparse.Namespace, data: DataProvider, model: torch.nn.Module,
                 sampler: TorchKacConstantSampler | None, reversal_fns: Reversal) -> float:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Extract the datasets from the loaders provided by the existing data pipeline
    cifar10_ds = data.get_dataset_for_full_eval()

    samples = sample(
        args=args,
        model=model,
        data=data,
        sampler=sampler,
        num_samples=args.eval_num_samples,
        reversal_fns=reversal_fns
    )
    generated_ds = Uint8Dataset(to_uint8_rgb(samples).cpu())

    print(f"Computing FID score via torch_fidelity using {args.eval_num_samples} samples...")
    
    metrics = torch_fidelity.calculate_metrics(
        input1=cifar10_ds,
        input2=generated_ds,
        batch_size=128,
        cuda=(device == 'cuda'),
        fid=True,
        verbose=False,
    )

    fid_score = metrics['frechet_inception_distance']
    
    return float(fid_score)


def eval_wrapper(args: argparse.Namespace, data: DataProvider, model: torch.nn.Module,
                 sampler: TorchKacConstantSampler | None, reversal_fns: Reversal):
    try:
        score = evaluate_fid(args=args, data=data, model=model, sampler=sampler, reversal_fns=reversal_fns)
        print(f"FID Score ({args.eval_num_samples} samples): {score:.4f}")
    except Exception as e:
        print(f"Evaluation failed: {e}")