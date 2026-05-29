import argparse

class Reversal():
    """This class implements the different reversal methods for the models, it is called by the sampling module as well as the distillation module
    """
    # ! If used in the distillation module, only explicit methods are to be specified, since distillation does not exactly with adaptive solvers.
    # TODO check that intuition again

    def __init__(self, args: argparse.Namespace):

        self.args = args

        if args.which == 'diffusion':
            pass

        elif args.which == 'kac':
            pass