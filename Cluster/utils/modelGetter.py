import argparse

from Cluster.networks.neuralNetworkOpenAI import UNetModel
from Cluster.utils.dataHandling import DataProvider

def model_getter(args: argparse.Namespace):
    data = DataProvider(args=args)

    if args.model_parameters == 'duong':
        # * Models the parameters of "2025 - Duong et al - Telegraphers"
        return UNetModel(
                image_size=data.data_dims.size, in_channels=data.data_dims.channels,
                out_channels=data.data_dims.channels,
                model_channels=128, num_res_blocks=2,
                attention_resolutions=(16,), num_heads=4,
                num_head_channels=64, channel_mult=(1, 2, 2, 2),
                dropout=0.1
            )

    elif args.model_parameters == 'han':
        # * Model parameters of "2026 - Han et al - DistillKac"
        return UNetModel(
                image_size=data.data_dims.size, in_channels=data.data_dims.channels,
                out_channels=data.data_dims.channels,
                model_channels=256, num_res_blocks=3,
                dropout=0.0, attention_resolutions=(16,8,4),
                num_heads=4, num_head_channels=64, channel_mult=(1, 2, 2, 3),
                use_scale_shift_norm=True, resblock_updown=True,
        )