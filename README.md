# cu_check

This repository presents a Proof of Concept that demonstrates how to use the Checkpoint API introduced in CUDA 12.8 together with CRIU. It includes implementation examples using widely adopted CUDA-based applications such as torchvision and Hugging Face Transformers. This work references the official [CUDA Checkpoint API documentation](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CHECKPOINT.html) and is inspired by the NVIDIA [cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint) repository.

## Prerequisites

- [cuda](https://developer.nvidia.com/cuda-toolkit)
- [criu](https://criu.org/Main_Page)
- python
  - [torch](https://pytorch.org/)
  - [torchvision](https://pytorch.org/vision/stable/index.html)
  - [transformers](https://huggingface.co/docs/transformers/index)

## Usage of `cu_check` tool

```bash
git clone https://github.com/suzusuzu/cu_check.git
cd cu_check

# build
gcc -I/usr/local/cuda-12.8/include cu_check.c -o cu_check -lcuda

# install
sudo mv cu_check /usr/local/bin

# Usage
cu_check state <pid>
cu_check lock <pid>
cu_check checkpoint <pid>
cu_check restore <pid>
cu_check unlock <pid>
```

## Experiments

### Pytorch Counter 

```bash
pip install torch
python torch_counter.py &
sleep 5
PID=$(pgrep -f 'python torch_counter.py')

# checkpoint
rm -rf tcnt && mkdir -p tcnt
cu_check lock $PID
cu_check checkpoint $PID
sudo criu dump -j -D tcnt -t $PID
du -sh tcnt

# restore
sudo criu restore -j -D tcnt &
while ! pgrep -f 'python torch_counter.py' > /dev/null 2>&1; do sleep 1; done
sudo cu_check restore $PID
sudo cu_check unlock $PID

sleep 5
kill -9 $PID
```

![](./img/torch_counter.gif)

### torchvision

```bash
git clone https://github.com/pytorch/examples.git
cd examples/imagenet/
pip install -r requirements.txt
python main.py -a resnet152 --dummy -j 0 &
sleep 20
PID=$(pgrep -f 'python main.py -a resnet152 --dummy -j 0')

# checkpoint
rm -rf resnet && mkdir -p resnet
cu_check lock $PID
cu_check checkpoint $PID
sudo criu dump -j -D resnet -t $PID
du -sh resnet

# restore
sudo criu restore -j -D resnet &
while ! pgrep -f 'python main.py -a resnet152 --dummy -j 0' > /dev/null 2>&1; do sleep 1; done
sudo cu_check restore $PID
sudo cu_check unlock $PID

sleep 20
sudo kill -9 $PID
```

![](./img/torchvision.gif)

### transformers

```bash
pip install transformers datasets accelerate
python train_bert.py &
sleep 60
PID=$(pgrep -f 'python train_bert.py')

# checkpoint
rm -rf bert && mkdir -p bert
cu_check lock $PID
cu_check checkpoint $PID
sudo criu dump -j -D bert -t $PID --tcp-established
du -sh bert

# restore
sudo criu restore -j -D bert --tcp-established &
while ! pgrep -f 'python train_bert.py' > /dev/null 2>&1; do sleep 1; done
sudo cu_check restore $PID
sudo cu_check unlock $PID

sleep 20
sudo kill -9 $PID
```

![](./img/transformers.gif)
