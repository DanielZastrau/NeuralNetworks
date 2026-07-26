# Framework for training, evaluating, sampling from and distilling from 4 different generative models

- *DDPM.py* implements the Diffusion model of "2020 - Ho et al - Denoising Diffusion Probabilistic Models" with the noise prediction loss and ancestral sampling

- *Salimans2022.py* implements the Diffuson Model as described in "2022 - Salimans Ho - Progressive Distillation" with the x0 prediction loss and the DDIM sampler

### Good to have

In VsCode I recommend to install the extension **Colorful Comments**, as I tried to provide hints and else throughout the code, which are a lot more visible if they can be discerned via color.

- ! comments denote something important
- TODO - comments denote something I have yet to resolve
- \* - comments denote a paper or source, where something is taken from