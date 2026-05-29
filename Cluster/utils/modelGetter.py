import argparse

from Cluster.neuralNetworkOpenAI import UNetModel
from Cluster.utils.dataHandling import DataProvider

def model_getter(args: argparse.Namespace):
    data = DataProvider(args=args)

    return UNetModel(
            image_size=data.data_dims.size, in_channels=data.data_dims.channels,
            out_channels=data.data_dims.channels,
            model_channels=128, num_res_blocks=2,
            attention_resolutions=[16], num_heads=4,
            num_head_channels=64, channel_mult=(1, 2, 2, 2)
        )