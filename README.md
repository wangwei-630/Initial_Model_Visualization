Initial Model Visualization
============================
Tested with PyTorch 1.7.0 on Ubuntu 20.04, CUDA 11.0.

Installation
============================
First, clone this repo and install dependencies:
```
conda create -n smplify-x python=3.7
conda activate smplify-x
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
cd Initial_Model_Visualization
pip install -r requirements.txt
```
Then prepare the SMPLX model files in smplx/models. The final directory structure should look like this:	<br>
```
smplx
└── models
	└── smplx
		├── SMPLX_FEMALE.npz
		├── SMPLX_FEMALE.pkl
        	├── SMPLX_MALE.npz
		├── SMPLX_MALE.pkl
        	├── SMPLX_NEUTRAL.npz
		└── SMPLX_NEUTRAL.pkl
└── smplifyx
	├── demo.py
	└── cmd_parser.py
└──README.md	
```

Usage
============================
For the visualization of the initial model, run:
```
python smplifyx/demo.py --model-folder models/smplx/SMPLX_NEUTRAL.npz --plot-joints=True --gender="neutral"
```

Citation	<br>
============================
```
@inproceedings{SMPL-X:2019,
  title = {Expressive Body Capture: 3D Hands, Face, and Body from a Single Image},
  author = {Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
  booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019}
}
```
