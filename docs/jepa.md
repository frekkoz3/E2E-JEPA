# The JEPA framework

The scope of this document is to provide an introduction to the framework of the *Joint-Embedding Predictive Architecture* (JEPA), covering at first the seminal ideas, and then the functional components of the state-of-the-art models. \
Finally, we will present our proposal for a new, stable and end-to-end trainable JEPA model.

---

## Contents
<!-- TOC -->
* [The JEPA framework](#the-jepa-framework)
  * [Contents](#contents)
  * [Self-Supervised Learning](#self-supervised-learning)
    * [Contrastive Learning](#contrastive-learning)
    * [Generative Algorithms](#generative-algorithms)
  * [JEPA architecture](#jepa-architecture)
  * [AV-JEPA](#av-jepa)
  * [References](#references)
<!-- TOC -->

---

## Self-Supervised Learning
The JEPA framework belongs to a class of algorithms under the wide umbrella of *self-supervised learning* (SSL). \
The idea behind SSL is to train models that generate output labels "intrinsically" from input data, so to reveal hidden relationships between data components or different views of data. \
Yann LeCun defined SSL as "a process to complete, or reconstruct, missing information", by, e.g., predicting any part of the input given any other part, or by predicting future from the past. 

It turns out that SSL algorithms are extremely able to solve not only the primary goal of the training process, but many other secondary (**proxy**) tasks: for example, models trained for image classification by using the SSL principle are able to predict images rotation or to predict colored versions given a gray-scale image.  

Under the umbrella of SSL fall lots of approaches that can be grouped into two main categories: contrastive methods and generative algorithms.

### Contrastive Methods
Contrastive learning is a self-supervised learning technique whose aim is to train models able to distinguish between similar (**positive**) and dissimilar (**negative**) data inputs. 

The general framework consists of three main steps. \
Raw data samples (the **anchor points**) are processed by data-augmentation methods; since the resulting new data will be in some way "similar" to the anchor. All the remaining data, instead, are considered negative examples. \
The examples are then encoded into a higher-dimensional space. A *projection head* then projects this embedded representation to a low-dimensional space. This step is done for the anchor and for all the positive and negative examples. \
Finally, in the third step, a **contrastive loss** computes the distances between similar and dissimilar data in this latent space.

The way positive and negative examples are chosen is extremely important. 
Usually, the number of negatives provided to the network is much higher than the number of positives. \
Since the number of negative examples is usually enormous to manage, two different solutions have been found. \
**SimCLR** employs a batch sampling strategy of size $N$. From the samples, $2N$ augmented pairs are generated. One pair is treated as a positive pair, while the other samples are considered negative. \
The main disadvantage of this approach is that it requires a high memory. 

Another approach consists into storing not raw data, but sample embeddings. In this way, the use of memory is extremely lower. The loss is computed w.r.t. to all these embeddings.
However, since the updates also regard the encoder, the stored embeddings will become soon useless. \
To prevent this, the **MoCo** architecture defines a momentum encoder, i.e., a copy of the data encoder. The momentum one is responsible for the embeddings of negatives, and its weights $\theta_k$ using an exponential smoothing w.r.t. the data encoder weights $\theta_q$:
$$
    \theta_k = \alpha \theta_k + (1-\alpha) \theta_q
$$

The definition of the loss is mostly case-dependent. \
Some common solutions are:
- **Noise-contrastive estimation (NCE)**
$$
    \mathcal{L}(\theta) = \sum_{i=1}^n \log \frac{p_\theta(y_i | x)}{p_\theta(y_i | x) + kp_k(y_i|x)} + \sum_{j=1}^m \log \frac{p_k(y_j | x)}{p_\theta(y_j | x) + kp_k(y_j|x)}
$$
    where $k$ is the number of negative samples, $p_k$ their distribution, $n$ the number of positives and $p_\theta$ their distribution. \
    This loss reduces the problem to a binary classification (similar/dissimilar).
- **Info Noise-contrastive estimation (NCE)**
$$
    \mathcal{L}(\theta) = -\log \frac{\exp(\frac{q \cdot k^+}{\tau})}{\sum_{i=0}^K \exp{\frac{q \cdot k_i}{\tau}} } 
$$
    where $q$ is the embedding of the ancor, $k^+$ is the embedding of the (unique) positive sample and $K$ is the set of the (multiple) negatives. \
    Its goal is to maximize the mutual information between positive samples pairs while minimizing the mutual information between negative samples pairs. \
    InfoNCE Loss has been used in the *MoCo* architecture.

- **Cross-entropy loss**
$$
    \mathcal{L} = -\log \frac{1}{\mathbf{B}} \sum_j^\mathbf{B} \sum_i^n y_i \log y_j
$$
    where $\mathbf{B}$ is the batch size.


Negative examples are not strictly required by contrastive learning algorithms, and they can be substituted by using **siamese networks**. \
Siamese networks aim to just maximize the similarity between two augmented versions of a single data point, while incorporating conditions and regularization so to prevent *collapsing* solutions, i.e., all the data are mapped to a single point. \
Usually, one network is kept fixed or updated more slowly (the *teacher*), while the other is continuously updated (the *student*). \
This kind of approach is called *self-distillation-based contrastive learning*.
Two notable algorithms are **SimSiam** and **BYOL**. \
SimSiam (Simple Siamese) exploits lightweight networks, consisting only of an encoder $f$ and a prediction head $h$, and optimizes a symmetric cosine similarity loss, defined as:
$$
    \mathcal{L} = \frac{1}{2} \bigg(-\frac{p_1}{||p_1||_2}\frac{z_2}{||z_2||_2} - \frac{z_1}{||z_1||_2}\frac{p_2}{||p_2||_2} \bigg)
$$
where $p_{(\cdot)} = h(z_{(\cdot)})$, $z_{(\cdot)} = f(x_{(\cdot)})$, and $x_1$ and $x_2$ are the two augmented data points. \
BYOL (Bootstrap-Your-Own-Latent), instead, employs two siamese networks updating the first at each training iteration, while using the other as the target. The updates of the second are swallowed always by using exponential moving average. 

Another completely different school of approach consists in learning decorrelated feature. \
This is the idea behind *feature-decorrelation-based contrastive learning*. \
**Barlow Twins** generates two views of a data point by sampling from a distribution of possible data augmentation techniques. The encoder returns two batches of embeddings $Z_A, Z_B$. The loss is defined so to minimize the redundancy between components, while maximizing the similarity between embedding vectors:
$$
    \mathcal{L} = \sum_i (1-C_{ii})^2 + \lambda \sum_i \sum_{j \neq i} C_{ij}^2,
$$
where $\lambda$ is a hyperparameter, and:
$$
C_{ij} = \frac{\sum_k z_{k,i}^A \cdot z_{k,j}^B }{\sqrt{\sum_k (z_{k, i}^A)^2}\sqrt{\sum_l (z_{l, j}^B)^2}}
$$
is the *cross-correlation matrix*. \
The very last approach we will cover is **VICReg** (Variance-Invariance-Covariance Regularization). \
In VICReg samples are generated exactly as in the BarlowTwin model. \
VICReg introduces a variance preservation term $\nu$, preventing collapse by penalizing the shrinkage of embedding vectors to zero:
$$
    \nu(Z_A) = \frac{1}{d} \sum_{j=1}^d \max \big(0, 1 - \sqrt{\mathbb{V}(z_j^A)} \big)
$$
The invariance criterion is simply a mean-squared distance:
$$
    \beta (Z_A, Z_B) = \frac{1}{n}\sum_{j=1}^n ||z_j^A - z_j^B||_2^2
$$
The covariance criterion is based on the covariance matrix $\mathbb{C}(Z)$:
$$
    \gamma(Z) = \frac{1}{d}\sum_{i\neq j}[\mathbb{C}(Z)]_{i,j}^2
$$
Finally, the last loss is defined as:
$$
    \mathcal{L} = \beta(Z_A, Z_B) + \lambda (\nu(Z_A) + \nu(Z_B)) + \kappa \big(\gamma(Z_A) + \gamma(Z_B) \big)
$$
Both regularization terms (variance and covariance) are here applied independently to each branch of the architecture. On the contrary, Barlow Twins exploits a unique cross-correlation matrix for both.    

### Generative Algorithms
Generative algorithms for SSL include a variety of models. \
The family that is most interesting for our project is the family of Masked (Image) Modelling, that includes **BEIT** (Bidirectional Encoder representation from Image Transformers) and **Masked AutoEncoders**. \
This family takes the name from the fact that a portion of the data, usually images or videos, is hidden, and the models are required to generate that missing chunk.

Both the architectures make use of vision transformers (ViT). \
BEIT introduces a MIM task for visual pretraining: breaks down the input image into visual tokens and then predicts a randomly masked subset of them. \
On the contrary, MAE tries to sparsify the image signals while using original pixels as its target. 

The **DINO** model bridges the gap between generative algorithms and contrastive learning. At its core, it is built on a Vision Transformer (ViT) and uses a "student-teacher" setup—meaning a student network learns by trying to mimic a teacher network. \
To keep the training stable regardless of how much data is processed at once (the mini-batch size), DINO adjusts the outputs using a moving average called "centering."
It then uses a temperature-scaled softmax function to turn these outputs into smooth probability distributions:
$$
    P_s(x) = \text{softmax}\left(\frac{f_{\theta_s}(x)}{\tau_s}\right)
$$
$$
    P_t(x) = \text{softmax}\left(\frac{f_{\theta_t}(x) - C}{\tau_t}\right)
$$
where $P_s$ is the student distribution, and $P_t$ the teacher one, given an augmented image view ($x$). \
Finally, the loss is the Cross-Entropy ($H$) between the teacher's prediction of one view ($x_2$) and the student's prediction of another view ($x_1$):
$$
    \mathcal{L} = - P_t(x_2) \log P_s(x_1)
$$
The discretization in DINO caused by the softmax can be interpreted as an online clustering
mechanism, where the last layer before the softmax contains the clustering prototypes
and its weight. As such, the output of the penultimate layer is clustered using the weights
of the last layer. 

---

## JEPA: the core ideas

---

## AV-JEPA

---

## Main References
Main papers:
- *Gui, Chen et al.*, 2024, ***A Survey on Self-Supervised Learning: Algorithms, Applications, and Future Trends***
- *Hu, Wang et al.*, 2020, ***A Comprehensive Survey on Contrastive Learning***
- *Balestriero*, 2023, ***A cookbook of Self-Supervised Learning***

Main videos:
- [AI Learns without labels](https://youtu.be/gVEr2cnDE_8?si=jFENjMPqFinjBfbe)
- [LeCun bet against LLM pt.1](https://youtu.be/kYkIdXwW2AE?si=E5MjYpMLiuvUFwzQ)
- [LeCun bet against LLM pt.2](https://youtu.be/v_jDvpEGTIg?si=JPk6rpWKYZV747Xs)

