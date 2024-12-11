# Pix2Pix - Conditional GAN for Segmented to Real Image Translation

In this project, we use Conditional GAN architecture for translating segmented images to real images in street view of Cityscapes dataset.

## Results

Here is some examples from validation set.

![result](https://www.kaggleusercontent.com/kf/212397397/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..0OppketrvAs9mtia7gtNAQ._EwE1uvyO--QpJick6Lsz1I1PJHe25iMcNKS1-OcgxWHhkt1nLe3ho3tzilM4t551biacz88ZlAUJG-beCPaBMBocy81SAESs0Beb6KKRTIzLUjzxJ0Mp0uB3dBtqpiwflZrODLdew8u8tnrnqV_TGi7qUGI5byfTjByeX1A2xgwWLPyM5pn3pt-HvyqSp4EHg5j9GXnMRJKV279foDcZhStI8mbZgT4jRyOG3BaKWk6MjnaNFEWy4csZezgTX-Oe5O2WmWZ_2AUTNN2W9beh_euzWR6ekOH8rywyNjbNV4VlWQuXvH3G-FxkuWhrWh7Zrjfnz16UYT7b1Qogs0-ic_NQTU_6l0X9xl27dff__-PuEH7GUOfIw3zcSBFJe_iLFYt6VkWj-qiqzPGWOG8x0i2J5YhRQPBsMO-W715eG1LFo7kl2WPpmdsYfpVpxkSAVdSzId3iY_VAMOlE9qbBq9-0_g3FIxlQEG8FW9Y9jGA5m0IGajX11iBkp2RGd68TuvRn122V3mElLx8BI-O5b48Da9mSOgNwbLrbAloIF_mMGy8Ni6G26CzHzZM-zefwRrLzklqsckNCbipU5M8_V4YH78HQri0Cjpjgj0gplQEvEEkknIgkf36IO_t8Omz8-kOx_x0cXCs6jaKwVzhrYJ64RVtM2OrII21CuJoXN8.7bGzkgJpj6DQ50rOKsYGAQ/__results___files/__results___29_0.png)

The loss of discriminator/generator for train/val for $50$ epochs.

![loss_curve](https://www.kaggleusercontent.com/kf/212397397/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..T0p30LuXyVH9jYEoXpN9zw.TVZ9vPC-s5h7tDmDMJJPjRuYYC0kOx8foYfyYjN3VwmOv6J2_rsiKX1Om6QxHpS9ZtcTG1rON1hVgEUal7i1n392A_Q8XMZupCTUsfYTCisOBzyPwvobQ20mydfN-IEMiMHKnfzR3xCINzlo656SA9bif5m9zZnGdcZheGWJtrVNwa6adO3uI9rzhvGLXyX0wDLbDZIHWjSBydBdSVOk0j3Uff3pt9zDA-9Hyp3JtYhVOsYKmN5bu4540NjEglNlbjvQYzvCeYL6Sq7Ov0Gm5RHTnZqoSX39p4SQkdq9hsNQkzgIAnJ51aKtChaEQ0TTzb8xIRigxp-DZO-TSxnVppYk4hCpqQS18FRKPI86g1VhQ1QE4L90L9gANfyBF3fsMcI82UQh3dceRaAiBP3cpj6urffgXrS_MjJgXSqX1SYeVPy6HxPZwNXBWjPLQ4-TN4Mlo-sgC5lr2lyaO6Z-y2SauPh6aGNrD46WYxfrQPjtRJEizrwP7JXqIrVA7DwdTHgqb_oVFx5Qisc2-nC9ptkOCJg6XYEuiFfNCiV0HwQMSXquEA1OtS5HVhvrmunvB4UVLdGaZbqgOvSoaHyCJigId0l7xmJyy7lZ_195DCIEvVyz1wGFB2lAsD5hgW9wthksdP1EcCdY6IdDonJ7rsGeSwaq7Ws4sslmirW2KS4.XDaaWenEpDCr14_Mv50CKg/__results___files/__results___27_0.png)

## Loss function

For more details see the [notebook](https://github.com/fraxea/pix2pix/pix2pix-conditional-gan-in-cityscapes.ipynb).

### Discriminator

The discriminator consist of a sequence of Convolution-BatchNorm-ReLU blocks. The generated-real recognition is done by classifying for $N\times N$ patches, then the final output of discriminator is the average of these results. This discriminator architecture is also called *PatchGAN*.

The discriminator loss is devided by $2$ to slow down the rate of learning for $D$.

$$
\mathcal{L}_D = -0.5 \big(\mathbb{E}_{x,y}[\log D(x,y)] + \mathbb{E}_{x,z}[\log (1-D(x, G(x,z)))]\big)
$$

### Generator

In this framework, the generator has *U-Net* architecture. Note that all convolution blocks have kernel size $4$ and ther is no pooling layer. Using random noise for $z$ is not effective in this case, so we use dropout with high probability $0.5$ to add diversity to our generator. So at inference time, we do **not turn off dropout**.

The effect of L-1 loss for generator is controlled by hyper-parameter $\lambda$. Also it is usual to optimize $-\log(D)$ instead of $\log(1-D)$, because of gradient reasons. So the following is the final generator loss function.

$$
\mathcal{L}_G = -\mathbb{E}_{x,z}[\log D(x, G(x,z))] + \lambda \mathbb{E}_{x,z,y}\big[|| y-G(x, z) ||_1\big]
$$

## Refrences

1. Official PyTorch Pix2Pix Implementation: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/
2. Original Paper of Pix2Pix: https://arxiv.org/pdf/1611.07004
3. Kaggle Notebook: https://www.kaggle.com/code/mohammadshafizd/pix2pix-conditional-gan-in-cityscapes
4. Medium Post: https://medium.com/@mohammadshafizd/pix2pix-in-cityscapes-dataset-e4d743b595b6
