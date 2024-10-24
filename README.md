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

## License Registration

After you apply for the SDK license from [Anygrasp](https://github.com/graspnet/anygrasp_sdk), put the license in ./grasp_detection/ and follow the [instruction](https://github.com/graspnet/anygrasp_sdk/blob/main/license_registration/README.md).

## Model Weight

After you get the [checkpoints](https://drive.google.com/file/d/1jNvqOOf_fR3SWkXuz8TAzcHH9x8gE8Et/view) of grasp_dectection, follow the [instruction](https://github.com/graspnet/anygrasp_sdk/tree/main/grasp_detection) and put model weights under ./grasp_detection/log/.

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

## Pipeline
Now you can run the code that uses AnyGrasp SDK & Grounded-Light-HQ-SAM.

1. Sampling
```
cd grasp_detection
python sample_realsense.py
# s - save
# q - exit
```
Then you can find the image (RGB+depth) captured in the "out" folder named by timestamps.

2. Query
```
cd Grounded-Segment-Anything/EfficientSAM
python grounded_light_hqsam.py
```
Then you will get the grounded detection and segmentation results under ./grasp_detection/out/.

3. Select
```
cd grasp_detection

# for mask
python select_mask_region.py

# for bounding box
python select_region.py

```

4. Grasp Pose Generation
```
cd grasp_detection
sh demo_real_query.sh
```
Finally, you will get the gripper_pose.json.

## Demo
You can run the code for example scene and the queried case - "bear" (under ./grasp_detection/out/1/).
