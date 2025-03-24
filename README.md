# Empilo

This repository provides the demo of the paper:

**"Empilo: Realizing Immersive Mobile 3D Video Conferencing through Parameterized Communication"**

---

## Getting Started

### 0. Python, PyTorch, and CUDA Requirements

To run this demo, ensure the following dependencies are installed in your environment:

- **Python**: 3.8 or higher (recommended: 3.10)
- **PyTorch**: 1.12 or higher (with CUDA support)
- **CUDA Toolkit**: 11.3 or higher (compatible with your PyTorch version)
- **Other Packages**: Listed in `requirements.txt`



### 1. Clone the Repository

```bash
git clone https://github.com/Empilo/empilo_demo.git
cd empilo_demo
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Pretrained Weights

Download Link: https://drive.google.com/drive/folders/1iLtnF0_p97CxOpXUBSAIwpAUgVkmiUGv?usp=sharing
```
empilo_demo/
└── pretrained/
    ├── deca_model.tar
    └── face_landmarker_v2_with_blendshapes.task
└── log/
    └── result/
        └── 0xxx-checkpoint.pth.tar
└── ...
```
Make sure all files are placed in the correct directories as shown above.

### 4. Run the Demo
```bash
python run.py \
  --config config/demo.yaml \
  --log_dir log \
  --ckp log/result/0xxx-checkpoint.pth.tar \
  --mode [test_xxx] \
  --precision [fp32 or fp16]

```

### 4. Available Modes
You can choose from the following demo modes using the `--mode` argument:
- `test_recon`: Reconstructs the original image. <br> → Outputs: [Ground Truth, Reconstruction]
- `test_style`: Transfers environment and lighting to the target face using a style source image. <br> → Outputs: [Expression Source, Synthesized, Style Source]
- `test_pose`: Modifies the face pose of the target image based on a pose source image. <br> → Outputs: [Expression Source, Synthesized, Pose Source]


### 5. Result
Latency (in `.txt`) and synthesized images (in `.png`) are saved in:
```
log/result/
├── latency_test_xxx.txt
    └── test_xxx-vis/epochxxx/
        ├── 0.png
        └── ...
