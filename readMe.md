# Framework for training, evaluating, sampling from and distilling from 4 different generative models

- *2020Ho.py* implements the Diffusion model of "2020 - Ho et al - Denoising Diffusion Probabilistic Models" with the noise prediction loss and ancestral sampling
- *2021Dhariwal.py* implements the improved Diffusion Model as describe in "2021 - Nichol & Dhariwal - Improved Denoising Diffusion Probabilistic Models"
- *2021Song.py* implements the DDPM ++ cont. model described in "2021 - Song et al. - Score based generative models through stochastic differential equations"
- *Salimans2022.py* implements the Diffuson Model as described in "2022 - Salimans & Ho - Progressive Distillation" with the x0 (or v) prediction loss and the DDIM sampler
- *2023SongCT.py* implements the standalone consistency model as described in "2023 - Song et al. - Consistency Models"

### Good to have

In VsCode I recommend to install the extension **Colorful Comments**, as I tried to provide hints and else throughout the code, which are a lot more visible if they can be discerned via color.

- ! comments denote something important
- TODO - comments denote something I have yet to resolve
- \* - comments denote a paper or source, where something is taken from