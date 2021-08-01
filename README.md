# ImageRetrieval

Accurate image retrieval is a core technology for shopping by picture taking, and also becomes a hotspot in the academia and industry. Here, we take a digital device image dataset in a real snap-to-shop scenario, provided in Huawei DIGIX Global AI Challenge. In a Content Based Image Retrieval (CBIR) System, the task is to retrieve similar images from a large database given a query image. The usual procedure is to extract some useful features from the query image, and retrieve images which have similar set of features. For this purpose, a suitable similarity measure is chosen, and images with high similarity scores are retrieved. Naturally the choice of these features play a very important role in the success of this system, and high level features are required to reduce the semantic gap.

Deep Hashing methods are most commonly used to implement content based image retrieval. If you do not know about image retrieval check out [this paper](https://arxiv.org/pdf/2006.05627.pdf).

Out of all the Deep Hashing methods we shall implement the [Hashnet](https://arxiv.org/pdf/1702.00758.pdf) and [Deep Cauchy Hashing](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-cauchy-hashing-cvpr18.pdf) for content based image retrieval.

# Requirements

Install the following with Python3 already installed:

* Pytorch (https://pytorch.org/)
* Numpy (https://numpy.org/)
* Pandas (https://pypi.org/project/pandas/)
* Pillow (PIL) (https://pypi.org/project/Pillow/)
* CUDA 11 (https://developer.nvidia.com/cuda-downloads) also make sure that you have GPU available on your system.

Other python libraries used are:

* Scikit-Learn
* Scikit-Image
* Matplotlib

# Datasets

The data used in the notebook can be found here.
 
[Train Data](https://www.kaggle.com/varenyambakshi/digixai-image-retrieval)

[Test Data](https://www.kaggle.com/varenyambakshi/digixalgoai)

# Network

Although there are many models available, we shall be using RESNET50. We will have broadly three types of layers namely

Convolution layers
Fully connected layers
Hashing layer
Instead of training the whole network at once we can use transfered learing and train few of the last layers. To further speed up training we can extract the intermediate feaures so that we do not have to pass the images everytime through the whole network. Instead we can just train on the extracted features.

