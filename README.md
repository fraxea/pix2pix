# Pix2Pix - Conditional GAN for Segmented to Real Image Translation

In this project, we use Conditional GAN architecture for translating segmented images to real images in street view of Cityscapes dataset.

## Results

Here is some examples from validation set.

![result](https://www.kaggleusercontent.com/kf/212539985/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..-PGWG99P_-r0bDw2KIBGvA.N-V9CTsqmKUnk35uQQfJZI4xhFJHJ4OQ0wNPqPDbLbIKns7FKWBR8iDVMTMhICuyh6UQIltmkpaaFo0Bkq7CaPobHNuqMguk23cqh7-FbjcdfFAcJItz2oZzSbZtRsXnSQIEEYXuKy5Lkg5qOIWfO3poeWpk1zLz4I_qCkti7MhYw3qVRuACRBESZsQ6Oa6OahgcOdz9nV-6cmJI4QW9rdw7pK57Ym42RI_ny1okD6TQghKobiELiazo0Zg0FG2iceMCqzWtelkQTvGOClE0rHcai1bBkyDeN5_bheBHyJJi66qDcwdXorqujiRhAqQHC2f9RCiYsr-fRc0RKhrz5yUeHPPDoItgIlSo2JRb50lY7Fe0_qOxTSSiux5Y0YchQzgJgcfOrDQThAlRBZKLYWph2FQ0UrvLZCkTB3FYJAD0ks6BehFB09euihfG5NowTSQeT6JZpqV749c_NRJPQ-ok0K3V39qGrQmqyQCcOY9qpn6GDlakPEQH-YO5mn6DWBgCmTMSQSV-tFvbr7FpdHvTsMCNXPEy1ZFzj94keL8wa5-BqOAoqFUbaq_KJJrJSqwErO1T4mYaLb4nZLLcAG-ZWIZanGwELy-qwvW0u4X0yV0eyRNfFGHYnKgnYDNJi1VGCFskchRr5FCMxy_PWFd_vk0DogAnhI-h0fTv4Yg.0ZvhMlzXb1dyAhzs0tOKGw/__results___files/__results___29_0.png)

The loss of discriminator/generator for train/val for 200 epochs.

![loss_curve](https://www.kaggleusercontent.com/kf/212539985/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..-PGWG99P_-r0bDw2KIBGvA.N-V9CTsqmKUnk35uQQfJZI4xhFJHJ4OQ0wNPqPDbLbIKns7FKWBR8iDVMTMhICuyh6UQIltmkpaaFo0Bkq7CaPobHNuqMguk23cqh7-FbjcdfFAcJItz2oZzSbZtRsXnSQIEEYXuKy5Lkg5qOIWfO3poeWpk1zLz4I_qCkti7MhYw3qVRuACRBESZsQ6Oa6OahgcOdz9nV-6cmJI4QW9rdw7pK57Ym42RI_ny1okD6TQghKobiELiazo0Zg0FG2iceMCqzWtelkQTvGOClE0rHcai1bBkyDeN5_bheBHyJJi66qDcwdXorqujiRhAqQHC2f9RCiYsr-fRc0RKhrz5yUeHPPDoItgIlSo2JRb50lY7Fe0_qOxTSSiux5Y0YchQzgJgcfOrDQThAlRBZKLYWph2FQ0UrvLZCkTB3FYJAD0ks6BehFB09euihfG5NowTSQeT6JZpqV749c_NRJPQ-ok0K3V39qGrQmqyQCcOY9qpn6GDlakPEQH-YO5mn6DWBgCmTMSQSV-tFvbr7FpdHvTsMCNXPEy1ZFzj94keL8wa5-BqOAoqFUbaq_KJJrJSqwErO1T4mYaLb4nZLLcAG-ZWIZanGwELy-qwvW0u4X0yV0eyRNfFGHYnKgnYDNJi1VGCFskchRr5FCMxy_PWFd_vk0DogAnhI-h0fTv4Yg.0ZvhMlzXb1dyAhzs0tOKGw/__results___files/__results___27_0.png)

## Loss function

For more details see the [notebook](https://github.com/fraxea/pix2pix/blob/main/pix2pix-conditional-gan-in-cityscapes.ipynb).

### Discriminator

The discriminator consist of a sequence of Convolution-BatchNorm-ReLU blocks. The generated-real recognition is done by classifying for patches, then the final output of discriminator is the average of these results. This discriminator architecture is also called *PatchGAN*.

The discriminator loss is devided by 2 to slow down the rate of learning for D.

### Generator

In this framework, the generator has *U-Net* architecture. Note that all convolution blocks have kernel size 4 and there is no pooling layer. Using random noise for z is not effective in this case, so we use dropout with high probability 0.5 to add diversity to our generator. So at inference time, we do **not turn off dropout**.

The effect of L-1 loss for generator is controlled by hyper-parameter lambda. Also it is usual to optimize -log(D) instead of log(1-D), because of gradient reasons.

## Refrences

1. Official PyTorch Pix2Pix Implementation: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/
2. Original Paper of Pix2Pix: https://arxiv.org/pdf/1611.07004
3. Kaggle Notebook: https://www.kaggle.com/code/mohammadshafizd/pix2pix-conditional-gan-in-cityscapes
4. Medium Post: https://medium.com/@mohammadshafizd/pix2pix-in-cityscapes-dataset-e4d743b595b6
