FROM paddlepaddle/paddle:latest-dev-cuda10.1-cudnn7-gcc82

   RUN pip3 install --upgrade pip
   RUN pip3 install paddlepaddle==2.0.0 -i https://mirror.baidu.com/pypi/simple

   RUN git clone https://github.com/PaddlePaddle/PaddleOCR.git
   WORKDIR ./
   RUN pip3 install -r requirements.txt

   WORKDIR /workspace