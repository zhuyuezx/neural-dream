# Neural Dream Video
This repo is a fork of [neural-dream](https://github.com/ProGamerGov/neural-dream) and can generate consistnet deep-dream style video based on input video.

This repo is mostly compatible with the original repo, so please refer to [README](https://github.com/ProGamerGov/neural-dream/blob/master/README.md) of parent repo for:
- Pytorch, CUDA, cuDNN installation
- Model download
- Parameter explanation
- FAQs

## Example of Transformation

**Video Input:**

https://github.com/zhuyuezx/neural-dream/assets/56620120/0d816ffa-4aae-4a46-b002-ca6b61b0d52a

**Video Output:**

https://github.com/zhuyuezx/neural-dream/assets/56620120/f9b5ba7e-44df-484f-b0c6-a2acdfacc2f8

## Additional Pre-requisites 

- opencv-python (`pip install opencv-python`)
- Pillow (`pip install Pillow`)
- Run command `python models/download_models.py` to download models

## Usage

**To make things light-weighted, you can refer to this [ipython notebook](https://colab.research.google.com/drive/1cGsbUHWECPCAjSV7_mKuS3t7IWC7kHSf?usp=sharing) to run code on Google Colab without the need of installing any dependencies. Feel free to add your own code snippets and play with the parameters!**

And here is the a basic example of how to run the code:
```
python neural_dream_video.py \
    -gpu 0 \
    -backend cudnn \
    -save_iter 0 \
    -image_size 960 \
    -num_octaves 2 \
    -learning_rate 1.5 \
    -num_iterations 1
```