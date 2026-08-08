# Base com CUDA 12.8 — primeiro toolkit a suportar Blackwell (sm_120 / RTX 5060 Ti)
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV KERAS_BACKEND=torch

# Sistema: Python 3.11 + ferramentas básicas
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    git curl ca-certificates && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    rm -rf /var/lib/apt/lists/*

# PyTorch com CUDA 12.8 + suporte sm_120 (Blackwell)
# Tenta primeiro stable cu128; se nao existir, usa nightly
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128 || \
    pip install --no-cache-dir --pre \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Resto das dependências
# tensorflow-cpu: só para tf.data pipeline (CPU); treino vai para GPU via torch
RUN pip install --no-cache-dir \
    "keras>=3.3" \
    tensorflow-cpu \
    mealpy==3.0.3 \
    scikit-learn \
    matplotlib \
    seaborn \
    pandas \
    jupyter \
    ipykernel \
    ipywidgets

WORKDIR /workspace
EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--allow-root", "--notebook-dir=/workspace", \
     "--NotebookApp.token=''", "--NotebookApp.password=''"]
