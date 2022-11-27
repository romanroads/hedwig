

pip install -U torch==1.4+cu100 torchvision==0.5+cu100 -f https://download.pytorch.org/whl/torch_stable.html && \
pip install cython pyyaml==5.1 && pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' \
&& python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.2.1'
