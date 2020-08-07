ARG PYVERSION="3.6.5"
ARG PYTORCH="1.5.1"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y \
    apt-utils \
    wget \
    curl \
    ca-certificates \
    sudo \
    git \
    unzip \
    htop \
    libglib2.0-0 \
    gnupg \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

RUN conda install cython -y && conda clean --all

RUN pip install -U pip
RUN conda install sklearn \
                  pandas \
                  tqdm \
                  Pillow \
                  ipython \
                  opencv-python \
                  albumentations

RUN pip install efficientnet_pytorch
RUN pip install pretrainedmodels

WORKDIR /workspace
COPY . /workspace

# miniconda and python
#ENV CONDA_AUTO_UPDATE_CONDA=false
#ENV PATH=/root/miniconda/bin:$PATH
#RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh \
# && chmod +x ~/miniconda.sh \
# && ~/miniconda.sh -b -p ~/miniconda \
# && rm ~/miniconda.sh \
# && conda update conda \
# && conda install -y python=${PYVERSION} \
# && conda clean -ya
#
## install pytorch-CUDA
#RUN conda install -y -c pytorch cudatoolkit=${CUDA} \
# && conda clean -ya

# requirements and apex install
# RUN pip install -r requirements.txt
#RUN git clone https://github.com/NVIDIA/apex
# RUN apex/python setup.py install --cuda_ext --cpp_ext
#RUN sed -i 's/check_cuda_torch_binary_vs_bare_metal(torch.utils.cpp_extension.CUDA_HOME)/pass/g' apex/setup.py
#RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"  ./apex

WORKDIR /workspace
COPY . /workspace

ENV PYTHONPATH="/workspace:$PYTHONPATH"

CMD ["/bin/bash"]
