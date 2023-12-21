# 'SimCLR' (Ting Chen et al., 2020) (partial) implementation from scratch in PyTorch
## Paper Reading
- [A Simple Framework for Contrastive Learning of Visual Representations](https://github.com/KimRass/SimCLR/blob/main/papers/a_simple_framework_for_contrastive_learning_of_visual_representations.pdf)
<!-- ## Introduction
- ***Composition of multiple data augmentation operations is crucial in defining the contrastive prediction tasks that yield effective representations.***
- ***Unsupervised contrastive learning benefits from larger batch sizes and longer training compared to its supervised counterpart.*** Like supervised learning, contrastive learning benefits from deeper and wider networks.
## Related Works
- Self-supervised learning (Representation learning)
    - ***Many approaches have relied on heuristics to design pretext tasks [1] [2] [3]***, which could limit the generality of the learned representations. Discriminative approaches based on contrastive learning in the latent space have recently shown great promise, achieving state-of-the-art results.
- Data augmentation
    - While data augmentation has been widely used in both supervised and unsupervised representation learning, ***it has not been considered as a systematic way to define the contrastive prediction task.***
## Methodology
- SimCLR learns representations by maximizing agreement between differently augmented views of the same data example via a contrastive loss in the latent space.
- ***A stochastic data augmentation module that transforms any given data example randomly resulting in two correlated views of the same example, denoted*** $\tilde{x}_{i}$ ***and*** $\tilde{x}_{j}$***, which we consider as a positive pair. In this work, we sequentially apply three simple augmentations: random cropping followed by resize back to the original size, random color distortions, and random Gaussian blur.***
- A neural network base encoder $f(\cdot)$ that extracts representation vectors from augmented data examples. Our framework allows various choices of the network architecture without any constraints. We adopt the commonly used ResNet to obtain $h_{i} = f(\tilde{x}_{i}) = \text{ResNet}(\tilde{x}_{i})$ where $h_{i} \in \mathbb{R}^{d}$ ***is the output after the average pooling layer.***
### Data Augmentation
- In our default pre-training setting (which is used to train our best models), we utilize random crop (with resize and random flip), random color distortion, and random Gaussian blur as the data augmentations.
- Random crop and resize to $224 \times 224$
    - The crop of random size (uniform from 0.08 to 1.0 in area) of the original size and a random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop is finally resized to the original size. Additionally, the random crop (with resize) is always followed by a random horizontal/left-to-right flip with 50% probability.
- Color distortion
    - Color distortion is composed by color jittering and color dropping. We find stronger color jittering usually helps, so we set a strength parameter.
- Gaussian blur
    - We blur the image 50% of the time using a Gaussian kernel. We randomly sample $\sigma \in [0.1, 2]$, and the kernel size is set to be 10% of the image height/width.
## Architecture
- Projection head
    - A small neural network projection head $g(\cdot)$ that maps representations to the space where contrastive loss is applied. We use a MLP with one hidden layer to obtain $z_{i} = g(h_{i}) = W^{(2)}\text{ReLU}(W^{(1)}h_{i})$. ***We find it beneficial to define the contrastive loss on*** $z_{i}$***’s rather than*** $h_{i}$’s ***.***
    - After training is completed, we through away the projection head $g(\cdot)$ and use encoder $f(\cdot)$ and representation $h$ for downstream tasks.
    - We use ResNet-50 as the base encoder network, and a 2-layer MLP projection head to project the representation to a 128-dimensional latent space.
## Training
- Default setting
    - We train at batch size 4096 for 100 epochs.
    - We use linear warmup for the first 10 epochs, and decay the learning rate with the cosine decay schedule without restarts [9].
### Loss Function
- Let $\text{sim(\textbf{u}, \textbf{v})}$ denote the cosine similarity between two vectors $\textbf{u}$ and $\textbf{v}$. Then the loss function for a positive pair of examples $(i, j)$ is defined as
$$\ell_{i, k} = -\log{\frac{\exp(sim(\textbf{z}_{i}, \textbf{z}_{j}) / \tau)}{\sum^{2N}_{k=1}\mathbb{1}_{[k \neq i]}\exp(\text{sim}(\textbf{z}_{i}, \textbf{z}_{k}) / \tau)}}$$
- where $\mathbb{1}_{[k \neq i]} \in \{0, 1\}$ is an indicator function evaluating to 1 iff $k \neq i$ and $\tau$ denotes a temperature parameter.
- For convenience, we term it ***NT-Xent (the normalized temperature-scaled cross entropy loss)***.
$$\mathcal{L} = \frac{1}{2N}\sum^{N}_{k = 1}[\ell(2k - 1, 2k) + \ell(2k, 2k - 1)]$$
<!-- - As the loss, we use NT-Xent, optimized using LARS with linear learning rate scaling (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6 . -->
- Sampling
    - We randomly sample a minibatch of $N$ examples and define the contrastive prediction task on pairs of augmented examples derived from the minibatch, resulting in $2N$ data points.
    - We do not sample negative examples explicitly. Instead, ***given a positive pair we treat the other*** $2(N − 1)$ ***augmented examples within a minibatch as negative examples.***
## Evaluation
- Dataset and metrics
    - Most of our study for unsupervised pretraining is done using the ImageNet ILSVRC-2012 dataset To evaluate the learned representations, ***we follow the widely used linear evaluation protocol, where a linear classifier is trained on top of the frozen base network, and test accuracy is used as a proxy for representation quality.***
    - Figure 6: Histograms of pixel intensities (over all channels) for different crops of two different images (one for each row)
        - <img src="https://user-images.githubusercontent.com/67457712/235917952-8695ecd0-a581-4c1e-a201-995567953e8c.png" width="400">
        - ***We conjecture that one serious issue when using only random cropping as data augmentation is that most patches from an image share a similar color distribution. Figure 6 shows that color histograms alone suffice to distinguish images. Neural nets may exploit this shortcut to solve the predictive task. Therefore, it is critical to compose cropping with color distortion in order to learn generalizable features.***
- Contrastive learning needs stronger data augmentation than supervised learning
    - Table 1: Top-1 accuracy of unsupervised ResNet-50 using linear evaluation and supervised ResNet-50 under varied color distor- tion strength
        - <img src="https://user-images.githubusercontent.com/67457712/235919073-3480a3f3-6db8-4254-b886-cb5f192ae576.png" width="400">
        - ***Stronger color augmentation substantially improves the linear evaluation of the learned unsupervised models.***
        - ***AutoAugment [5], a sophisticated augmentation policy found using supervised learning, does not work better than simple cropping + (stronger) color distortion.***
        - ***When training supervised models with the same set of augmentations, we observe that stronger color augmentation does not improve or even hurts their performance. Thus, our experiments show that unsupervised contrastive learning benefits from stronger (color) data augmentation than supervised learning.***
        - ***Although previous work has reported that data augmentation is useful for self-supervised learning [1] [6] [7], we show that data augmentation that does not yield accuracy benefits for supervised learning can still help considerably with contrastive learning.***
        - "1 (+Blur)" is our default data augmentation policy.
- Unsupervised contrastive learning benefits (more) from bigger models
    - Figure 7: Linear evaluation of models with varied depth and width
        - <img src="https://user-images.githubusercontent.com/67457712/235921419-4b4180e5-9941-403a-b589-9fa04c18fd2e.png" width="400">
        - Blue dots: Ours trained for 100 epochs
        - Red stars: Ours trained for 1000 epochs
        - Green crosses: Supervised ResNets trained for 90 epochs. Training longer does not improve this model.
        - We find the ***gap between supervised models and linear classifiers trained on unsupervised models shrinks as the model size increases, suggesting that unsupervised learning benefits more from bigger models than its supervised counterpart.***
- Contrastive learning benefits (more) from larger batch sizes and longer training
    - Figure 9: Linear evaluation of ResNet-50 trained with different batch size and epochs
        - <img src="https://user-images.githubusercontent.com/67457712/235922884-5ca86041-85f4-4697-8e9d-36efaa5d9766.png" width="480">
        - ***We find that, when the number of training epochs is small (e.g. 100 epochs), larger batch sizes have a significant advantage over the smaller ones. With more training steps/epochs, the gaps between different batch sizes decrease or disappear***, provided the batches are randomly resampled.
        - In contrast to supervised learning [8], ***in contrastive learning, larger batch sizes provide more negative examples, facilitating convergence (i.e. taking fewer epochs and steps for a given accuracy). Training longer also provides more negative examples, improving the results.***
        - Each bar is a single run from scratch.
## References
- [1] [Unsupervised Visual Representation Learning by Context Prediction, 2015](https://arxiv.org/pdf/1505.05192.pdf)
- [2] [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles, 2016](https://arxiv.org/pdf/1603.09246.pdf)
- [3] [Unsupervised Representation Learning by Predicting Image Rotations, 2018](https://arxiv.org/pdf/1803.07728.pdf)
- [4] [Discriminative Unsupervised Feature Learning with Exemplar Convolutional Neural Networks, 2014](https://arxiv.org/pdf/1406.6909.pdf)
- [5] [AutoAugment: Learning Augmentation Policies from Data, 2019](https://arxiv.org/pdf/1805.09501.pdf)
- [6] [Learning Representations by Maximizing Mutual Information Across Views, 2019](https://arxiv.org/pdf/1906.00910.pdf)
- [7] [Data-Efficient Image Recognition with Contrastive Predictive Coding, 2019](https://arxiv.org/pdf/1905.09272.pdf)
- [8] [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour, 2017](https://arxiv.org/pdf/1706.02677.pdf) -->
