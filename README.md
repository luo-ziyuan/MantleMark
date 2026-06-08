# MantleMark

Official code for **MantleMark: Migrating Watermarks from Multi-View Images to Radiance Fields via Frequency Modulation**.

Paper: [IEEE Xplore](https://ieeexplore.ieee.org/document/11263898)

## Structure

- `NeRF/`: watermark migration for NeRF and TensoRF.
- `3DGS/`: watermark migration for 3D Gaussian Splatting.

## Installation

Use a CUDA-enabled Linux environment for training and rendering.

For NeRF/TensoRF:

```bash
cd NeRF
pip install -r requirements.txt
bash scripts/install_ext.sh
```

If you use `--tcnn`, also install tiny-cuda-nn:

```bash
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

For 3D Gaussian Splatting:

```bash
cd 3DGS
bash requirements.sh
```

## NeRF Example

Place a Blender-style NeRF dataset at `NeRF/data/nerf_synthetic/<scene>`.

Train an original NeRF:

```bash
cd NeRF
python main_nerf_watermark.py data/nerf_synthetic/lego \
  --workspace outputs/lego_original \
  -O --bound 1.0 --scale 0.8 --dt_gamma 0 --tcnn
```

Create watermarked multi-view images:

```bash
python w_embedder_with_pose_with_GS.py \
  --input_data_path data/nerf_synthetic/lego \
  --output_data_path data/nerf_synthetic/watermark_tifs/lego/lego_watermarked_70_100 \
  --min_f 70 --max_f 100 --watermark_strength 1.0
```

Train on the watermarked views:

```bash
python main_nerf_watermark.py data/nerf_synthetic/watermark_tifs/lego/lego_watermarked_70_100 \
  --workspace outputs/lego_watermarked_70_100 \
  -O --bound 1.0 --scale 0.8 --dt_gamma 0 --tcnn
```

Detect the watermark:

```bash
python w_detection_from_image_save_image_MF.py \
  --original_results outputs/lego_original \
  --watermarked_results outputs/lego_watermarked_70_100 \
  --watermarked_dataset data/nerf_synthetic/watermark_tifs/lego/lego_watermarked_70_100 \
  --n_test 200
```

The default detector is `--detector band_zero`. The previous matched-filter detector is available with `--detector matched_filter`.

For TensoRF, use `main_tensoRF.py` instead of `main_nerf_watermark.py`.

## 3DGS Example

Prepare a COLMAP or Blender-compatible scene, then train:

```bash
cd 3DGS
python train.py -s /path/to/scene -m outputs/scene_3dgs
```

The watermarking and detection scripts mirror the NeRF workflow:

```bash
python w_embedder_with_pose_with_GS.py --input_data_path /path/to/scene --output_data_path /path/to/watermarked_scene
python w_detection_from_image_save_image_MF.py --original_results /path/to/original_outputs --watermarked_results /path/to/watermarked_outputs --watermarked_dataset /path/to/watermarked_scene --n_test 200
```

## Notes

Datasets and checkpoints are not included.

Base code:
- `NeRF/`: adapted from [torch-ngp](https://github.com/ashawkey/torch-ngp).
- `3DGS/`: adapted from [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting).

The `NeRF/` and `3DGS/` folders keep their original upstream licenses.
