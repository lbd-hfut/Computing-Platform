FROM nvidia/cuda:12.6.2-base-ubuntu24.04
COPY . /Code
RUN apt update && apt install -y libopencv-dev
RUN pip install torch numpy opencv-python matplotlib scipy

