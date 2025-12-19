# Lab6 Transformer Quantization Report

1. **What's the difference between SmoothQuant and the method in Lab3?** 10%

    Lab 3 (Standard MinMax/Percentile Quantization) typically quantizes activations and weights directly. This approach struggles with models like ViTs that have severe activation outliers, as the quantization range must accommodate the outliers, crushing the precision of normal values. 
    **SmoothQuant** introduces a mathematical transformation *before* quantization to **migrate the quantization difficulty** from activations to weights. It divides activations by a smoothing factor $s$ (smoothing outliers) and multiplies weights by $s$ (making them spikier). Since weights are generally more robust to quantization than outlier-heavy activations, this results in better accuracy for INT8 quantization.

1. **When applying SmoothQuant, where do activation values get divided by the smooth factor?** 10%

    The activations are divided by the smooth factor implicitly within the **LayerNorm** parameters before the Linear layers. In the `smooth_ln_fcs` function, the code modifies the LayerNorm weights and biases:
    `ln.weight.div_(scales)`
    `ln.bias.div_(scales)`
    Since LayerNorm output is $Y = \frac{X-\mu}{\sigma}\gamma + \beta$, scaling $\gamma$ and $\beta$ by $1/s$ effectively scales the output activation $X$ by $1/s$.

1. **How is the smooth factor being calculated?** 10%

    The smooth factor $s$ is calculated per-channel based on the maximum absolute values of the activations ($|X|$) and weights ($|W|$), balanced by a strength parameter $\alpha$ (typically 0.5):
    $s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}$
    In the code: `scales = (act_max.pow(alpha) / weight_max.pow(1 - alpha)).clamp(min=1e-5)`

1. **What's the difference between ViT-S and CNN models when doing quantization?** 10%

    **ViT-S (Vision Transformers)** are significantly harder to quantize than **CNNs** due to the presence of **severe systematic activation outliers**. These outliers degrade the effective resolution of quantization for the majority of the data. **CNNs** typically have more uniform or Gaussian-like activation distributions (e.g., after ReLU) and are naturally more robust to direct quantization. SmoothQuant is specifically designed to address the outlier challenge inherent in ViTs.

1. **What's your observation on the visualization of weight and activation values distribution?** 10%

    The "Before vs After" 3D surface plots (Block 11) confirm the migration of magnitude:

    - **Activations:** The smoothed surface appears **"Flatter"** (reduced peaks and outliers), making it easier to quantize.
    - **Weights:** The smoothed surface appears **"Spikier"** (larger magnitude range), as they have absorbed the scale from the activations.

    This validates that SmoothQuant effectively shifts the quantization burden from the sensitive activations to the robust weights.