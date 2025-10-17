FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get -y update 
RUN apt-get upgrade -y 
RUN apt-get install -y libprotobuf-dev protobuf-compiler
RUN apt-get install -y cmake
RUN apt-get install -y git
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get -y update 
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libopencv-dev

RUN pip install albumentations==1.1.0 dlib==19.24.0 python-box==7.1.1 imgaug==0.4.0 && \
    pip install imutils==0.5.4 ipython numpy==1.23.3 opencv-python==4.5.1.48 && \
    pip install pandas==1.3.5 Pillow==9.3.0 scikit-image==0.19.3 scipy==1.9.3 simplejson && \
    pip install tensorboardX==2.5.1 tqdm PyYAML
RUN pip install retinaface_pytorch
RUN pip install efficientnet_pytorch
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install timm
RUN mkdir /.cache
RUN chmod -R a+rwX /.cache

WORKDIR /workspace
