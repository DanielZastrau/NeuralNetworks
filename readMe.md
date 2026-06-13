# Framework for training, evaluating, sampling from and distilling from 4 different generative models

# Features of the Software

- We use `torch.amp`. **amp** stands for **automatic mixed precision**. In particular, two concepts of it: `torch.amp.autocast` and `torch.amp.GradScaler`.
    - `torch.amp.autocast`: is a context manager within which we call the model. It dynamically detects which operations are stable and temporarily casts the fp32 masterweights to fp16 to speed up calculations. Unstable operations like Softmax, LayerNorm and the final loss calculation remain in fp32.
    - `torch.amp.GradScaler`: Is an entity which handles scaling of values to allow for highest precision within fp16. It maintains a current scale factor **S**. After loss calculation the fp32 loss is multiplied by S, and the backward pass / the gradient calculation is done on the scaled scaled loss. Each model layer / operation calculates its gradient in the precision it used in the forward pass. This is the exact mechanism which enables higher precision gradient calculation in fp16. They are subsequently cast to fp32 and unscaled (divided by S) to allow the optimizer to work with the true mathematical values. Lastly, it evaluates the gradients. If they were stable for 2k iterations it ups the scale (allowing for higher precision), it they were unstable it reduces the scale. Scales of up to ~2^23 are not unusual
    - `optimizer`: The optimizer maintains an fp32 copy of the models weights. If any gradient is **NaN** or **Inf** the step is skipped. Skipping one iteration is acceptable in any amount of iterations usually encountered in training deep neural networks.
    - `Reason`: In training the core entities are the gradients. These gradients can typically become very small, which often leads to underflow issues in fp16 precision, which subsequently corrupts the model. There are multiple workarounds:
        1. **training in fp32**, which yields double the memory footprint and usually makes training much slower, since it disallowes the use of modern tensor cores, which are often specialized for fp16/bf16 operations.
        2. **automatic scaling** as described above
        3. **bf16** a standard developed by google which has the same footprint as fp16 (2Bytes) but uses a 8 exponent bits instead of the fp16 5 exponent bits


### Good to have

In VsCode I recommend to install the extension **Colorful Comments**, as I tried to provide hints and else throughout the code, which are a lot more visible if they can be discerned via color.

- ! comments denote something important
- TODO - comments denote something I have yet to resolve
- \* - comments denote a paper or source, where something is taken from