# ACQuA: Anomaly Classification with Quasi-Attractors 

## Overview
We present a novel framework for classifying abnormalities from cardiac signals. Our model, ACQuA is capable of detecting murmurs from PCG signals and classifying arrhythmias from ECG signals. 
We evaluate our model on the CinC/Physionet 2017 Challenge data which tasks practitioners to classifying single channel ECG signals into one of four classes: Normal, AFib, Other Arrhythmia and too noisy to be classified. 
Additionally, we test compete in the CinC/Physionet 2022 Challenge data which evaluates models on their ability to detect heart murmurs from PCG signals. 

Our approach is to cast a signal classification problem to an image classification problem by inputting quasi-attractor images into a custom ResNet Model. A quasi-attractor is the approximate set of steady states of the underlying set of 
differential equations that govern the morphology of the signal. We project the signal into a high-dimensional quasi-attractor using a time-delay embedding, then compress the quasi-attractor into a 2D image using PCA. 

For more details, please access our pre-print below:

https://www.medrxiv.org/content/10.1101/2022.08.31.22279436v1.full.pdf+html



## ACQuA For Murmur Detection 

## ACQuA for Cardiac Arrhythmia Classification 

## Useful links

- [George B. Moody 2022 Challenge](https://physionetchallenges.org/2022/)
- [CinC/Physionet 2017 Challenge](https://www.physionet.org/content/challenge-2017/1.0.0/)
