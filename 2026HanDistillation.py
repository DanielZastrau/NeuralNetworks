import os
import argparse

import torch

from Cluster.utils.dataHandling import DataProvider

class Distillation():

    def __init__(self, student_steps: int = 100, teacher_substeps: int = 100):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.student_steps = student_steps
        self.teacher_substeps = teacher_substeps

        self.data = DataProvider(argparse.Namespace(
            training_batch_size = 128, eval_num_samples = 50_000,
        ))

    def integrator(self): 
        #! performs a euler step from t to t - dt

    def routine(self):

        self.teacher.eval()
        self.student.train()

        train_dl, test_dl = self.data.get_datasets_for_training()
        
