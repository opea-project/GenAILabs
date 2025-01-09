# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Use the same python version with ray
FROM python:3.10.14

ARG HF_TOKEN

ENV HF_TOKEN=$HF_TOKEN

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

COPY comps /home/user/comps

ENV SQFT_PATH=/home/user/comps/sqft

RUN chown -R user $SQFT_PATH

USER user

ENV PATH=$PATH:/home/user/.local/bin

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install --no-cache-dir intel-extension-for-pytorch && \
    python -m pip install --no-cache-dir oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/

WORKDIR $SQFT_PATH

# Clone and install GenAIComps
RUN git clone https://github.com/opea-project/GenAIComps $SQFT_PATH/GenAIComps && \
cd $SQFT_PATH/GenAIComps && \
git checkout v1.1 && \
pip install -e . && \
pip install --no-cache-dir -r $SQFT_PATH/GenAIComps/comps/finetuning/requirements.txt

# Clone and install peft with patch
RUN git clone https://github.com/huggingface/peft.git $SQFT_PATH/peft && \
    cd $SQFT_PATH/peft && \
    git checkout v0.10.0 && \
    git apply --ignore-space-change --ignore-whitespace $SQFT_PATH/patches/peft-v0.10.0.patch && \
    pip install -e .

ENV PYTHONPATH=$PYTHONPATH:$SQFT_PATH/GenAIComps

RUN echo PKGPATH=$(python3 -c "import pkg_resources; print(pkg_resources.get_distribution('oneccl-bind-pt').location)") >> run.sh && \
    echo 'export LD_LIBRARY_PATH=$PKGPATH/oneccl_bindings_for_pytorch/opt/mpi/lib/:$LD_LIBRARY_PATH' >> run.sh && \
    echo 'source $PKGPATH/oneccl_bindings_for_pytorch/env/setvars.sh' >> run.sh && \
    echo ray start --head --dashboard-host=0.0.0.0 >> run.sh && \
    echo export RAY_ADDRESS=http://localhost:8265 >> run.sh && \
    echo python sqft_service.py >> run.sh

CMD bash run.sh
