import os
import argparse
import copy
import importlib

from typing import Protocol, Optional

import torch
import torch_fidelity

from Cluster.utils.dataHandling import DataProvider
from Cluster.utils.dataAugmentation import KarrasAugmentationPipeline
from Cluster.utils.uint8_utils import Uint8Dataset, to_uint8_rgb

class GenerativeModel(Protocol):
    """This exists to satisfy the type checker for the Distillation class"""

    # ! needed on all
    curr_dir: str
    S: int
    model: torch.nn.Module
    score_save_path: str

    def noisify(self, t: torch.Tensor, x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def get_model(self) -> None:
        ...

    def sample(self, model: torch.nn.Module, amount: int) -> torch.Tensor:
        ...

    # used by duong and karras
    def model_fn(self, model: torch.nn.Module, t: torch.Tensor | float, x: torch.Tensor,
                 aug_cond: Optional[torch.Tensor]) -> torch.Tensor:
        """This should be outputting the predicted velocity field."""
        ...

    # ! specific to the individual models
        
    augmentation_pipeline: KarrasAugmentationPipeline    # used by Karras, duong augmented, and mmd augmented

    # used by karras
    def get_karras_schedule(self):
        ...

    # used by karras
    def get_karras_schedule_betw_tensors(self, S: int, t_min: torch.Tensor, t_max: torch.Tensor) -> torch.Tensor:
        ...

    # used by song
    def v(self, t: torch.Tensor, x: torch.Tensor, model: torch.nn.Module, graph: bool = False) -> torch.Tensor:
        ...

    # used by karras
    def sigma(self, t: float | torch.Tensor) -> float | torch.Tensor:
        ...

    # used by karras
    def weight(self, t: torch.Tensor) -> torch.Tensor:
        ...

class Distillation():

    def __init__(self, which: str, model: GenerativeModel, student: torch.nn.Module, teacher: torch.nn.Module,
                 student_steps: int = 100, teacher_substeps: int = 100, loss: str = 'original',
                 score_checking: bool = False):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data = DataProvider(argparse.Namespace(
            training_batch_size = 128, eval_num_samples = 50_000,
        ))

        self.which = which
        self.model = model
        self.teacher = teacher
        self.student = student

        self.student_steps = student_steps
        self.teacher_substeps = teacher_substeps

        self.loss = loss

        self.epsilon = 1e-5

        self.I = 50_000 * teacher_substeps
        self.lr = 1e-4
        self.lr_warmup = int(self.I * 0.05)

        self.student_save_path = os.path.join(self.model.curr_dir, f'{self.student_steps}student.pth')

        self.score_checking = score_checking

        print(f'Distilling    {self.model.score_save_path},  to a    {student_steps}  student with    \
              {teacher_substeps}  many teacher substeps and saving to    {self.student_save_path}.  Loss target:    {loss}')

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

        if 'Karras' in self.which:
            self.model.S = self.student_steps
            self.t_endpoints = torch.tensor(self.model.get_karras_schedule(), device=self.device, dtype=torch.float32)
        else:
            self.t_endpoints = torch.linspace(1, self.epsilon + dt_student, self.student_steps, device=self.device, dtype=torch.float32)

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

            if 'Karras' in self.which:
                loss = self.update_karras(x0=x0)

            else:    # Duong simple, Mmd, Song, Zastrau

                if self.which == '2025Mmd' or self.which == '2026Zastrau':
                    x0_aug, aug_cond = self.model.augmentation_pipeline(x0)
                else:
                    x0_aug, aug_cond = x0, None

                n = torch.randint(0, self.student_steps, (x0.shape[0],), device=self.device)
                t = self.t_endpoints[n]

                xt, _ = self.model.noisify(t=t, x0=x0_aug)

                #? Euler stepping with the teacher
                with torch.no_grad():
                    xtarget = xt.clone()
                    for step in range(self.teacher_substeps):
                        tprime = t - step * dt_teacher

                        v_teacher = self.model.model_fn(model=self.teacher, t=tprime, x=xtarget, aug_cond=aug_cond)
                        xtarget = xtarget - dt_teacher * v_teacher

                #? Euler stepping with the student
                xpred = xt.clone()
                v_student = self.model.model_fn(model=self.student, t=t, x=xpred, aug_cond=aug_cond)
                xpred = xpred - dt_student * v_student


                if self.loss == 'original':
                    loss = torch.nn.functional.mse_loss(xpred, xtarget)
                else:    # self.loss == 'vspace'
                    #* This is similar to the target calculation of Salimans and Ho
                    vtarget = (xt - xtarget) / dt_student

                    loss = torch.nn.functional.mse_loss(vtarget, v_student)

            print(f'Loss:  {loss.item()}')
            loss.backward()

            grad_norm = torch.sqrt(sum(p.grad.data.norm() ** 2 for p in self.student.parameters() if p.grad is not None))
            print(f'Grad norm: {grad_norm.item()}')

            self.optim.step()
            self.scheduler.step()
            self.ema.update_parameters(self.student)

            if (iteration + 1) % 10_000 == 0:
                self.eval(model=self.ema)

            if (iteration + 1) % 20_000 == 0:
                uncompiled_model = getattr(self.ema.module, "_orig_mod", self.ema.module)
                torch.save(uncompiled_model.state_dict(), self.student_save_path)
                print(f"saved student model to:  {self.student_save_path}    at iteration {iteration}")


    def update_karras(self, x0: torch.Tensor):

        x0_aug, aug_cond = self.model.augmentation_pipeline(x0)

        # we have the student karras schedule from which we first need to sample endpoints
        n = torch.randint(0, self.student_steps, (x0.shape[0],), device=self.device)

        # Karras schedule is decreasing. Therefore, ti > tip
        student_ti = self.t_endpoints[n]
        student_tip = self.t_endpoints[n + 1]
        dt = student_ti - student_tip

        xt, _ = self.model.noisify(student_ti, x0=x0_aug)

        # [B, teacher_substeps + 1]
        teacher_endpoints: torch.Tensor = self.model.get_karras_schedule_betw_tensors(S = self.teacher_substeps, t_min=student_tip, t_max=student_ti)
        with torch.no_grad():
            xtarget = xt.clone()

            for i in range(self.teacher_substeps):
            
                ti = teacher_endpoints[:, i]
                tip = teacher_endpoints[:, i + 1]

                sigma_ti = self.model.sigma(ti)
                sigma_tip = self.model.sigma(tip)

                assert isinstance(sigma_ti, torch.Tensor)
                assert isinstance(sigma_tip, torch.Tensor)    # This exists to satisfy my type checker that sigma_tip is in fact a tensor

                sigma_ti_bc = sigma_ti.view(-1, *([1] * (x0.dim() - 1)))
                sigma_tip_bc = sigma_tip.view(-1, *([1] * (x0.dim() - 1)))

                diff = tip - ti
                diff_bc = diff.view(-1, *([1] * (x0.dim() - 1)))

                # ? pfode evaluation at the current timestep
                pred_ti = self.model.model_fn(model=self.teacher, x=xtarget, t=sigma_ti, aug_cond=aug_cond)
                dt = ( 1 / sigma_ti_bc) * xtarget - (1 / sigma_ti_bc) * pred_ti

                # ? Euler step
                x_intermediate = xtarget + diff_bc * dt

                # ? Second order correction
                mask = (sigma_tip != 0).float().view(-1, *([1] * (x0.dim() - 1)))
                if mask.any():
                    safe_sig_tip = torch.where(sigma_tip_bc == 0, torch.ones_like(sigma_tip_bc, device=self.device), sigma_tip_bc)
                    
                    pred_tip = self.model.model_fn(model=self.teacher, t=safe_sig_tip.flatten(), x=x_intermediate, aug_cond=aug_cond)
                    dtprime = (1 / safe_sig_tip) * x_intermediate - (1 / safe_sig_tip) * pred_tip
                    
                    xtarget = xtarget + diff_bc * (0.5 * dt + 0.5 * dtprime) * mask + diff_bc * dt * (1 - mask)
                else:
                    xtarget = x_intermediate

        # 6. Student Integration (Single Heun Step)
        xpred = xt.clone()

        student_sigma_ti = self.model.sigma(t=student_ti)
        student_sigma_tip = self.model.sigma(t=student_tip)

        assert isinstance(student_sigma_ti, torch.Tensor)
        assert isinstance(student_sigma_tip, torch.Tensor)

        student_sigma_ti_bc = student_sigma_ti.view(-1, *([1] * (x0.dim() - 1)))
        student_sigma_tip_bc = student_sigma_tip.view(-1, *([1] * (x0.dim() - 1)))

        student_diff = student_tip - student_ti
        student_diff_bc = student_diff.view(-1, *([1] * (x0.dim() - 1)))
        
        # Evaluate student at student_ti
        pred_student_ti = self.model.model_fn(model=self.student, t=student_ti, x=xpred, aug_cond=aug_cond)
        dt_student = (1 / student_sigma_ti_bc) * xpred - (1 / student_sigma_ti_bc) * pred_student_ti
        
        x_inter_student = xpred + student_diff_bc * dt_student
        
        mask_student = (student_tip != 0).float().view(-1, *([1] * (x0.dim() - 1)))
        if mask_student.any():
            safe_sig_tip_stud = torch.where(student_sigma_tip_bc == 0, torch.ones_like(student_sigma_tip_bc, device=self.device), student_sigma_tip_bc)
            
            pred_student_tip = self.model.model_fn(model=self.student, t=safe_sig_tip_stud.flatten(), x=x_inter_student, aug_cond=aug_cond)
            dtprime_student = (1 / safe_sig_tip_stud) * x_inter_student - (1 / safe_sig_tip_stud) * pred_student_tip
            
            xpred = xpred + student_diff_bc * (0.5 * dt_student + 0.5 * dtprime_student) * mask_student + student_diff_bc * dt_student * (1 - mask_student)
        else:
            xpred = x_inter_student

        weight = self.model.weight(student_ti).view(-1, *([1] * (x0.dim() - 1)))
        return (weight * torch.nn.functional.mse_loss(xpred, xtarget, reduction='none')).mean()

    def eval(self, model: torch.nn.Module | None = None):

        #! Final fid evaluation on 50k samples
        
        if model is None:
            # Load the best model
            self.model.get_model()
            self.model.model.load_state_dict(torch.load(self.student_save_path, map_location=self.device))

            eval_model = self.model.model
        else:
            eval_model = model

        self.model.S = self.student_steps

        print(f'Evaluating    {self.student_save_path}.')

        eval_ds = self.data.get_dataset_for_full_eval()

        eval_model.eval()
        samples = self.model.sample(model=eval_model, amount=50_000)

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
    want to rename them all.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--what', type=str, default='full', choices=['full', 'distill', 'eval'])
    parser.add_argument('--which', type=str, required=True, choices=['2026Duong', '2026Zastrau', '2025Mmd', '2022Karras', '2021Song'])
    parser.add_argument('--student-steps', type=int, required=True)
    parser.add_argument('--teacher-substeps', type=int, required=True)
    parser.add_argument('--loss', type=str, default='original', choices=['original', 'vspace'])
    parser.add_argument('--score_checking', type=bool, action='store_true')
    args = parser.parse_args()

    if args.which == '2026Duong':
        duong_module = importlib.import_module('2026Duong')
        model = duong_module.Kac(which='simple', schedule='uniform', integrator='euler', S=100, load_teacher=True)
        teacher = model.model
        student = copy.deepcopy(teacher)

    elif args.which == '2026Zastrau':
        zastrau_module = importlib.import_module('2026Zastrau')
        model = zastrau_module.DSBFM(which='augmented', load_teacher=True)
        teacher = model.model
        student = copy.deepcopy(teacher)

    elif args.which == '2025Mmd':
        mmd_module = importlib.import_module('2025MMD')
        model = mmd_module.MMD(size='small', data_augmentation=True, load_teacher=True)
        teacher = model.model
        student = copy.deepcopy(teacher)

    elif args.which == '2022Karras':
        karras_module = importlib.import_module('2022Karras')
        model = karras_module.EDM(S=18, load_teacher=True)
        teacher = model.model
        student = copy.deepcopy(teacher)

    elif args.which == '2021Song':
        song_module = importlib.import_module('2021Song')
        model = song_module.DDPMppCont(load_teacher=True)
        teacher = model.model
        student = copy.deepcopy(teacher)

    Distillery = Distillation(which=args.which, model=model, student=student, teacher=teacher,
                              student_steps=args.student_steps, teacher_substeps=args.teacher_substeps,
                              loss=args.loss, score_checking=args.score_checking)
    
    if args.what == 'full' or args.what == 'distill':
        Distillery.routine()

    if args.what == 'full' or args.what == 'eval':
        Distillery.eval()