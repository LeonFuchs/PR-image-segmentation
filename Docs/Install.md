# Installation

* Install uv : 
    * On Windows : ```powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"```
    * On Mac or Linux : ```wget -qO- https://astral.sh/uv/install.sh | sh```

* Download this repository's files and extract them on your computer
* Setup the virtual environment :
    * Navigate to the ```script/ ``` directory
    * Create a virtual environment : ```uv venv PR-is --python 3.12```
        * You may replace ```PR-is```in this and other commands with another name as you see please
    * Activate the virtual environment : ```PR-is\Scripts\activate```
    * Install python libraries : ```uv pip install numpy opencv-python pyqt6 matplotlib tensorflow keras tiffile scipy```

You can now run the application 