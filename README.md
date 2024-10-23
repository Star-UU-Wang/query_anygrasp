<img src="https://user-images.githubusercontent.com/12446953/208367719-4ef7922f-4001-41f7-aa9f-076e462d1325.png" width="60%">

# Query_AnyGrasp
AnyGrasp SDK & Grounded-Light-HQ-SAM for queried grasp detection & tracking.

## Requirements
- Python 3.8/3.9/3.10
- PyTorch 1.7.1 with CUDA 11.0+
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) v0.5.4


## Installation for Anygrasp
1. Follow MinkowskiEngine [instructions](https://github.com/NVIDIA/MinkowskiEngine#anaconda) to install [Anaconda](https://www.anaconda.com/), cudatoolkit, Pytorch and MinkowskiEngine. **Note that you need ``export MAX_JOBS=2;`` before ``pip install`` if you are running on an laptop due to [this issue](https://github.com/NVIDIA/MinkowskiEngine/issues/228)**. If PyTorch reports a compatibility issue during program execution, you can re-install PyTorch via Pip instead of Anaconda.

2. Install other requirements from Pip.
```bash
    pip install -r requirements.txt
```

3. Install ``pointnet2`` module.
```bash
    cd pointnet2
    python setup.py install
```

## Installation for Grounded-Light-HQSAM
Please follow [Grounded-SAM's Installation](https://github.com/IDEA-Research/Grounded-Segment-Anything) to create an additional conda env called sam:
```
conda create -n sam python=3.9
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda-11.8/
```
Then install torch:
```
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```
Then:
```
python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
pip install --upgrade diffusers[torch]
git submodule update --init --recursive
cd grounded-sam-osx && bash install.sh
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel
```
Download pretrained Light-HQSAM weight [here](https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth) and put it in ./Grounded-Segment-Anything/EfficientSAM/, then download the pretrained groundingdino weights:
```
cd Grounded-Segment-Anything

# download the pretrained groundingdino-swin-tiny model
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

## License Registration
   
Due to the IP issue, currently we can only release the SDK library file of AnyGrasp in a licensed manner. Please get the feature id of your machine and fill in the [form](https://forms.gle/XVV3Eip8njTYJEBo6) to apply for the license. See [license_registration/README.md](license_registration/README.md) for details. **If you are interested in code implementation, you can refer to our [baseline version of network](https://github.com/graspnet/graspnet-baseline), or a third-party implementation of our [GSNet](https://github.com/graspnet/graspness_unofficial).**

We usually reply in 2 work days. If you do not receive the reply in 2 days, **please check the spam folder.**


## Demo Code
Now you can run your code that uses AnyGrasp SDK. See [grasp_detection](grasp_detection) and [grasp_tracking](grasp_tracking) for details.

1. Sampling

