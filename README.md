# Memristor-based feature learning for pattern classification

**Pytorch implementation of "Memristor-based feature learning for pattern classification"**

## Data

Download the data from Baidu Netdisk. https://pan.baidu.com/s/1uAgodlD973s6H5xg89d4bw Extraction Code: 4321.

## Implementation of DDK Network

![DDK Network.png](https://github.com/gllspeed/DDK/tree/main/image-20241014162015113.png)

## Usage

### Train

Five application experiments were conducted.  You can run the experiments by entering the project folder and run `python train.py`or `sh ./run.sh`   You will need to modify the path of the data in the code to your own.

- **Train example in SITW:**

`cd DDK\Audio_recognition\DDK+FC\adapt_convNet_pytorch_SITW1`

`python train.py`

## Result

- Comparision between DDK Network and Convolutional Neural Network in SITW, AGNews, MNIST, UCF and ModelNet.

  ​              **Accuracy Parameter Operation for validation data**

![DDK Result.png](https://github.com/gllspeed/DDK/tree/main/image-20241014161531341.png)
