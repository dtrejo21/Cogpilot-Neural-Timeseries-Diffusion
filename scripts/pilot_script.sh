#!/bin/sh

python3 ../ntd/train_diffusion_model.py \
    dataset=cogpilot \
    diffusion=diffusion_linear_200 \
    diffusion_kernel=white_noise \
    optimizer.lr=0.0001 \
    optimizer.num_epochs=500 \
    network=ada_conv_cogpilot \
    optimizer=base_optimizer \
    optimizer.lr=0.0004 \
    +experiments/generate_samples=generate_samples

echo "DONE"