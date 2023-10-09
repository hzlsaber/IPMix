# IPMix

The office repository of "IPMix: Label-Preserving Data Augmentation Method for Training Robust Classifiers"
by Zhenglin Huang,Xianan Bao,Na Zhang,Qingqi Zhang,Xiaomei Tu,Biao Wu, and Xi Yang

 IPMix is a simple data augmentation approach to improve robustness without hurting clean accuracy. IPMix integrates three levels of data augmentation (image-level, patch-level, and pixel-level) into a coherent and label-preserving technique to increase the diversity of training data with limited computational overhead. 
![](https://github.com/hzlsaber/IPMix/blob/main/images/IPMix.png)

# Abstract

Data augmentation has been proven effective for training high-accuracy convolutional neural network classifiers by preventing overfitting. However, building deep
neural networks in real-world scenarios requires not only high accuracy on clean data but also robustness when data distributions shift. While prior methods have
proposed that there is a trade-off between accuracy and robustness, we propose IPMix, a simple data augmentation approach to improve robustness without hurting clean accuracy. IPMix integrates three levels of data augmentation (image-level, patch-level, and pixel-level) into a coherent and label-preserving technique to increase the diversity of training data with limited computational overhead. To further improve the robustness, IPMix introduces structural complexity at different levels to generate more diverse images and adopts the random mixing method for multi-scale information fusion. Experiments demonstrate that IPMix outperforms state-of-the-art corruption robustness on CIFAR-C and ImageNet-C. In addition, we show that IPMix also significantly improves the other safety measures, including robustness to adversarial perturbations, calibration, prediction consistency, and anomaly detection, achieving state-of-the-art or comparable results on several benchmarks, including ImageNet-R, ImageNet-A, and ImageNet-O.

![](https://github.com/hzlsaber/IPMix/blob/main/images/performance.png)
# Usage

CIFAR:
```
python cifar.py \
  --dataset <cifar10 or cifar100> \
  --data-path <path/to/cifar and cifar-c> \
  --mixing-set <path/to/mixing_set> \
  --model wrn
```

ImageNet:
```
python imagenet.py \
  --data-standard <path/to/imagenet_train> \
  --data-val <path/to/imagenet_val> \
  --imagenet-r-dir <path/to/imagenet_r> \
  --imagenet-c-dir <path/to/imagenet_c> \
  --mixing-set <path/to/mixing_set> \
  --num-classes 1000 \
```

# Mixing-set

You can download the IPMix set used in the paper here. 

Since IPMix is insensitive to mixing sets change, you can use any unlabeled synthetic pictures (e.g., diffusion images) freely. If you tried IPMix with other mixing sets, please feel free to let me know the results, regardless of positive or negative. Many thanks!


# Pretrained Models

Weights for CIFAR-10/100 classifier (Wrn40-4/Wrn28-10/ResNet-18/ResNeXt-29) trained with IPMix are available here.

Weights for a ResNet-50 classifier trained with IPMix for 180 epoches is available here.

# Acknowledgments

Our code is developed based on [AugMix](https://github.com/google-research/augmix/tree/master) and [PixMix](https://github.com/andyzoujm/pixmix).

