# Installation

* Open a powershell window
* Install uv : ```powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"```
* Create a virtual environment : ```uv venv PR-image-segmentation --python 3.12```
* Activate the virtual environment : ```PR-image-segmentation\Scripts\activate```
* Install python libraries : ```uv pip install numpy opencv-python pyqt6 matplotlib tensorflow keras tiffile scipy```

