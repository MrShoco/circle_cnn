# Circle finder
Fully Convolutional Neural Network trained to find circle in noisy image.

## General comments
Most of the code is inside the **core/** folder.
I tried to keep the provided main.py with minimum changes.

In general, the approach is to run FCNN network on the image and predict a grid
of probabilities, relative coordinates and radiuses. Then aggregate that 
information to find the best center and radius assumption. The model easily 
achieves > 0.999@iou>0.7 score.

Model structure is in the **core/circle_net.py** Training log is in the 
**output.txt**. In order to run the code, use tips below.

## Build
To build container:
```bash
docker build -f Dockerfile -t circle-net .
```

With gpu support:
```bash
docker build -f Dockerfile.nvidia -t circle-net-gpu .
```

## Train
To train CNN run:
```bash
docker run -it -v $PWD:/tf circle-net python core/train.py
```

With gpu support:
```bash
docker run --gpus=all -it -v $PWD:/tf circle-net python core/train.py
```

## Inference
To get score for latest checkpoint:
```bash
docker run -it -v $PWD:/tf circle-net python main.py
```
After 30 epochs model achieves > 0.999 score. Latest checkpoints are in the 
**/checkpoints** folder.
