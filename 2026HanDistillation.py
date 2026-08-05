import os
import argparse
import copy
import importlib

from typing import Protocol, Optional

import torch
import torch_fidelity

from Cluster.utils.dataHandling import DataProvider
from Cluster.utils.uint8_utils import Uint8Dataset, to_uint8_rgb

class GenerativeModel(Protocol):
    """This exists to satisfy the type checker for the Distillation class"""

    curr_dir: str
    S: int
    model: torch.nn.Module
    score_save_path: str

    def noisify(self, t: torch.Tensor, x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def model_fn(self, model: torch.nn.Module, t: torch.Tensor, x: torch.Tensor, aug_cond: Optional[torch.Tensor]) -> torch.Tensor:
        ...

    def get_model(self) -> None:
        ...

    def sample(self, model: torch.nn.Module, amount: int) -> torch.Tensor:
        ...    

class Distillation():

    def __init__(self, model: GenerativeModel, student: torch.nn.Module, teacher: torch.nn.Module, student_steps: int = 100, teacher_substeps: int = 100):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data = DataProvider(argparse.Namespace(
            training_batch_size = 128, eval_num_samples = 50_000,
        ))

        self.student = student
        self.teacher = teacher
        self.model = model

        self.student_steps = student_steps
        self.teacher_substeps = teacher_substeps

        self.epsilon = 1e-5

        self.I = 50_000 * teacher_substeps
        self.lr = 1e-4
        self.lr_warmup = int(self.I * 0.05)

        self.student_save_path = os.path.join(self.model.curr_dir, f'{self.student_steps}student.pth')

        print(f'Distilling    {self.model.score_save_path},  to a    {student_steps}  student with    {teacher_substeps}  many teacher substeps and saving to    {}')

    def get_ema(self):

        self.ema = torch.optim.swa_utils.AveragedModel(self.student, device=self.device, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay=0.9999))

    def get_optim(self):

        self.optim = torch.optim.Adam(self.student.parameters(), lr=self.lr) 
        self.scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optim,
                                                  start_factor=0.2,
                                                  end_factor=1.0,
                                                  total_iters=self.lr_warmup)

    def routine(self):

        self.teacher.eval()
        self.student.train()

        self.get_ema()
        self.get_optim()

        dt_student = (1 - self.epsilon) / self.student_steps
        dt_teacher = dt_student / self.teacher_substeps
        t_endpoints = torch.linspace(1, self.epsilon + dt_student, self.student_steps, device=self.device, dtype=torch.float32)

        train_dl, _ = self.data.get_datasets_for_training()
        train_iter = iter(train_dl)
        for iteration in range(self.I):
            if (iteration + 1) % 1_000 == 0:
                print(f'>>>>>>>>>>    iteration    {iteration}    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

            try:
                x0, _ = next(train_iter)
                x0 = x0.to(self.device, dtype=torch.float32)

            except StopIteration:
                train_iter = iter(train_dl)
                x0, _ = next(train_iter)
                x0 = x0.to(self.device, dtype=torch.float32)

            self.optim.zero_grad()

            self.student.train()
            self.ema.train()

            n = torch.randint(0, self.student_steps, (x0.shape[0],), device=self.device)
            t = t_endpoints[n]

            xt, _ = self.model.noisify(t=t, x0=x0)

            #? Euler stepping with the teacher
            with torch.no_grad():
                xtarget = xt.clone()
                for step in range(self.teacher_substeps):
                    tprime = t - step * dt_teacher
                    xtarget = xtarget - dt_teacher * self.model.model_fn(model=self.teacher, t=tprime, x=xtarget, aug_cond=None)

            #? Euler stepping with the student
            xpred = xt.clone()
            xpred = xpred - dt_student * self.model.model_fn(model=self.student, t=t, x=xpred, aug_cond=None)

            loss = torch.nn.functional.mse_loss(xpred, xtarget)
            print(f'Loss:  {loss.item()}')
            loss.backward()

            grad_norm = torch.sqrt(sum(p.grad.data.norm() ** 2 for p in self.student.parameters() if p.grad is not None))
            print(f'Grad norm: {grad_norm.item()}')

            self.optim.step()
            self.scheduler.step()
            self.ema.update_parameters(self.student)

            if (iteration + 1) % 20_000 == 0:
                uncompiled_model = getattr(self.ema.module, "_orig_mod", self.ema.module)
                torch.save(uncompiled_model.state_dict(), self.student_save_path)
                print(f"saved student model to:  {self.student_save_path}    at iteration {iteration}")

    def eval(self):

        #! Final fid evaluation on 50k samples
        
        # Load the best model
        self.model.get_model()
        self.model.model.load_state_dict(torch.load(self.student_save_path, map_location=self.device))
        self.model.S = self.student_steps

        print(f'Evaluating    {self.student_save_path}.')

        eval_ds = self.data.get_dataset_for_full_eval()

        self.model.model.eval()
        samples = self.model.sample(model=self.model.model, amount=50_000)

        gen_ds = Uint8Dataset(to_uint8_rgb(samples, self.data))

        metrics = torch_fidelity.calculate_metrics(
            input1=eval_ds,
            input2=gen_ds,
            batch_size=256,
            fid=True,
            cuda=(('cuda' if torch.cuda.is_available() else 'cpu') == 'cuda'),
            verbose=False,
        )
        ema_score = metrics['frechet_inception_distance']

        print(f"Tested the ema model. FID Score (50_000 samples): {ema_score:.4f}")


if __name__=='__main__':
    """Using importlib instead of normal imports, because I already committed to a naming convention of the files and I don't
    want to rename them all."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--what', type=str, default='full', choices=['full', 'distill', 'eval'])
    parser.add_argument('--which', type=str, required=True, choices=['2026Duong', '2025Mmd', '2026Zastrau', '2022Karras'])
    parser.add_argument('--student-steps', type=int, required=True)
    parser.add_argument('--teacher-substeps', type=int, required=True)
    args = parser.parse_args()

    if args.which == '2026Duong':
        duong_module = importlib.import_module('2026Duong')
        model = duong_module.Kac(which='simple', schedule='uniform', integrator='euler', S=100, load_teacher=True)
        teacher = model.model
        student = copy.deepcopy(teacher)

        Distillery = Distillation(model=model, student=student, teacher=teacher, student_steps=args.student_steps, teacher_substeps=args.teacher_substeps)

    if args.what == 'full' or args.what == 'train':
        Distillery.routine()

    if args.what == 'full' or args.what == 'eval':
        Distillery.eval()