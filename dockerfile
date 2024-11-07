FROM tensorflow/tensorflow:2.9.0-gpu

RUN apt update
RUN apt install ffmpeg libsm6 -y
RUN apt install nano

RUN pip install --upgrade pip
RUN pip install opencv-python
RUN pip install torch
RUN pip install torchvision
RUN pip install mean_average_precision
RUN pip install progressbar
RUN pip install scikit-learn
RUN pip install tqdm
RUN pip install pandas
RUN pip install Pillow
RUN pip install matplotlib
RUN pip install scikit-multilearn
RUN pip install tensorflow-addons
RUN pip install pycocotools
RUN pip install opentsne
RUN pip install examples

RUN echo 'alias ll="ls -l"' >> ~/.bashrc
