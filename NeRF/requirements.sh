
yes | pip install -r requirements.txt
yes | pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

yes | pip uninstall opencv-python
yes | pip uninstall opencv-contrib-python
yes | pip uninstall opencv-contrib-python-headless

yes | pip3 install opencv-contrib-python==4.5.5.62
yes | pip install opencv-python==4.5.5.64

apt-get update && apt-get install -y libgl1

yes | pip install imageio[ffmpeg]
yes | pip install imageio[pyav]

# cd ../3DGS

# yes | pip install imageio[ffmpeg]
# yes | pip install imageio[pyav]
# yes | pip install submodules/diff-gaussian-rasterization
# yes | pip install submodules/simple-knn
# yes | pip install plyfile
# yes | pip install protobuf==3.20.*
# yes | pip install Pillow==9.5.0