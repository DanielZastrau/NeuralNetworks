# Implementations of several generative modeling frameworks

- *2020Ho.py* implements the Diffusion model of "2020 - Ho et al - Denoising Diffusion Probabilistic Models" with the noise prediction loss and ancestral sampling
- *2021Dhariwal.py* implements the improved Diffusion Model as describe in "2021 - Nichol & Dhariwal - Improved Denoising Diffusion Probabilistic Models"
- *2021Song.py* implements the DDPM ++ cont. model described in "2021 - Song et al. - Score based generative models through stochastic differential equations"
- *Salimans2022.py* implements the Diffuson Model as described in "2022 - Salimans & Ho - Progressive Distillation" with the x0 (or v) prediction loss and the DDIM sampler
    [06.08.26] - does not work
- *2023SongCT.py* implements the standalone consistency model as described in "2023 - Song et al. - Consistency Models"
    [06.08.26] - does not work
- *2025MMD.py* implements the MMD Gradient flow toward a uniform distribution as described in "2025 - Chemseddine et al - Adapting Noise to Data"
- *2026Duong.py* implements the Kac generative model as described in "2026 - Duong et al - Telegraphers Generative Model via Kac Flows"
- *2026Han.py* implements the endpoint distillation algorithm as described in "2026 - Han et al - DistillKac"
- *2026Zastrau.py* implements the DSB-FM algorithm as described in my master thesis

# Run the models

All models are written such that simply running the files without any arguments begins training and evaluating a base configuration of the respective models.

# Related repositories

For the logs of the training runs see [Link](https://github.com/DanielZastrau/Logs)