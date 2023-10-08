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

**Video Output(lr = 0.25):**

https://github.com/zhuyuezx/neural-dream/assets/56620120/f9b5ba7e-44df-484f-b0c6-a2acdfacc2f8

## Additional Pre-requisites 

- opencv-python (`pip install opencv-python`)
- Pillow (`pip install Pillow`)
- Run command `python models/download_models.py` to download models

## Usage

**To make things light-weighted, you can refer to  [google_colab_demo.ipynb](https://colab.research.google.com/drive/1cGsbUHWECPCAjSV7_mKuS3t7IWC7kHSf?usp=sharing) to run code on Google Colab without the need of installing any dependencies. Feel free to add your own code snippets and play with the parameters!**

And here is the a basic example of the python command, and according to the parameter defaults, it will generate a full-length video with 960x540 resolution using `videos/balcony_view.mp4` as input, and save the output video to `videos_out/balcony_view_out.mp4`:
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
And here is the result (stronger deep-dream effect as learning rate is 1.5):

https://github.com/zhuyuezx/neural-dream/assets/56620120/60d3adaf-a6ec-4894-86b1-42720bd8858a

## Essential Parameters Explanation

Those you should indicate in every run:
- `input_video` : Path to input video, default is `videos/balcony_view.mp4`
- `output_video` : Path to output video, default is `videos_out/balcony_view_out.mp4`
- `gpu` : 'c' for CPU, and '0' for GPU, default is '0'
- `backend` : choices from ['nn', 'cudnn', 'mkl', 'mkldnn', 'openmp', 'mkl,cudnn', 'cudnn,mkl'], and default is 'nn', and `cudnn` is recommended on colab

Those you may use frequently:
- `image_size` : Width of the frames extracted from input video, default is 960, and height will be calculated automatically with ratio `16:9`
- `output_size` : Output video size, will be `image_size` if unspecified
- `num_octaves` : Number of octaves with 2 as default, and please adjust according to your `image_size` in case it's too big
- `learning_rate` : Learning rate of gradient ascent, default is 1.5, larger value will result in stronger deep-dream effect
- `num_iterations` : Number of iterations for each octave, default is 10, while larger value results in stronger deep-dream effect
- `dream_layers` : comma separated layer names to apply deep-dream, with `inception_4d/3x3_reduce` as default, and you can add more layers to focus on different layers (please refer to `models/googlenet/bvlc_googlenet.py` for all layer names)
  
Those you may use occasionally:
- `in_dir` & `out_dir` : Directories for storing extracted frames and output frames, default is `frames_in` and `frames_out` (will be created automatically if not exist)
- `start_idx` & `end_idx` : Start and end index of frames to be extracted from input video to apply neural dream, default is `[first frame ~ last frame]`
- `save_iter` : Save output video every `save_iter` frames, default is 0 (only save the last frame)
- `predefined_start` : Start with image `start.png` instead of the first frame of input video, default is 0. You can also use this in case there's interruption in the middle of the process, then use the last saved frame as `start.png` and set `predefined_start` to the index of the last saved frame
- `channels` : Focus only on indicated channel(s) inside `dream_layers`, indicate multiple channels using comma separated value. In default, all channels will be targed
- `just_extract` : Only extract frames from input video, and skip neural dream process, default is 0
- `skip_merging` : Extract frames and apply neural dream, but skip merging frames into output video, default is 0

For any unmentioned parameters in code, please refer to [original README](https://github.com/ProGamerGov/neural-dream/blob/master/README.md#usage).