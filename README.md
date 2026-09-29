# Classification-of-Images-for-Building-Damage-Assessment
This project is centered around using deep learning technologies for a civil engineering application, namely recognition of structural damage from images. Damage surveys currently require teams of domain experts to visually inspect buildings to determine their safety, which is slow and subjective. Our objective is to automate the process using computer vision, in this project we use a supervised learning approach over 8 different recognition tasks: scene level, damage state, spalling condition, material type, collapse mode, component type, damage level, and damage type. The model will be constructed from 8 different CNN models and each image will be passed through the 8 recognition tasks to see which class in each recognition task it applies to. The challenge will be running all 8 recognition tasks to their fullest and handling the individual errors that might show up in each model along with the load on the PC to run these models consecutively.

This project aims to replace the current process, simplify the process and add consistency to the work done on analyzing structural damage and we believe it is possible for this type of model to start taking over this type of work on a larger level, especially in cases of natural disasters where we need a quick and accurate analysis of the structural damage at a time where resources are spread thin, this type of project creates an unlimited resource to pull from in those situations.

# Introduction
Ensuring the proper performance of all elements in a structure is a priority for designers and users. In most cases, continuous monitoring can detect damages at an early stage can prevent potential accidents and catastrophes that result from inadequate inspection or damages to the evaluation process. Structural health monitoring (SHM) involves the use of continuous monitoring using sensors that are permanently attached to the structure, together with algorithms related to the damage-identification process.

Collapses of civil infrastructures strike public opinion more and more often. They are generally due to either structural deterioration or modified working conditions with respect to the design ones. The main challenge of structural health monitoring is to increase the safety level of ageing structures by detecting, locating and quantifying the presence and the development of damages, possibly in real-time

However, visual inspections—whose frequencies are usually determined by the importance and the age of the structure—are still the workhorse in this field, even if they are rarely able to provide a quantitative estimate of structural damages. Therefore, it is evident why recent advances in sensing technologies and signal processing, coupled to the increased availability of computing power, are creating huge expectations in the development of robust and continuous SHM systems.

Structural health monitoring (SHM) and rapid damage assessment after natural hazards and disasters have become an important focus in civil engineering. Moreover, structural response records and images as the data media play an increasing role in nowadays data explosion epoch. Meanwhile, artificial intelligence (AI) and machine learning (ML) technologies are developing rapidly, especially in applications of deep learning (DL) in computer vision, which made giant progress in recent years. 

In addition, the objective of implementation of ML and DL is to make computers perform labor-intensive repetitive tasks and also learn from past experiences. Nowadays, structural damage recognition using images is one of the important topics in vision-based SHM and structural reconnaissance, which greatly relies on human visual inspection and experience. However, several recent non-DL studies are addressing issues related to relatively tedious manual efforts.

Thus, following this trend, it is timely to implement the state-of-art DL technologies in civil engineering applications and evaluate its potential benefits.
Computer vision is a field of artificial intelligence that trains the computer to interpret and understand the visual world. Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects and then react to what they see. In deep learning a convolutional neural network is a class of artificial neural network most commonly applied to analyze visual imagery, CNN has been at the heart of spectacular recent advances in deep learning and is better than traditional computer vision and machine learning approaches due to many reasons including its depth of architecture and it no longer needing low level features or feature engineering, so CNNs use relatively little preprocessing compared to other image classification algorithms.

With such strong advances in image classification we have the chance to use it for structural health monitoring, so far only a few researches or applications of CNN exist in post disaster reconnaissance or SHM, civil engineering applications have not fully benefited from the data driven computer science or computer vision technologies, and so our aim is to combine the two fields and have a program that civil engineering can benefit from greatly.

The field of civil engineering needs more studies geared towards machine learning to get the full benefit from that field and this project is a step in that direction.
In general, our objectives for this project are as follows: 

1.	Gather labelled and unlabeled data for structural health monitoring and post disaster reconnaissance for the purpose of applying machine learning algorithms
2.	Create 8 different recognition tasks covering thousands of images each to be classified into multiple classes for each recognition task
3.	Make an effective, consistent, and time efficient program capable of classifying structural damage on multiple levels and ready for regular use in structural health monitoring 

# Methodology
Each of the 8 recognition tasks gets its own classifier, trained on that task's images from the Φ-Net dataset.

**Data.** The images are stored as numpy arrays. They are opened with `mmap_mode='r'` and read from disk one batch at a time through a `tf.data` pipeline, so a task never has to fit in memory. 25% of each training set is held out for validation, split so every class keeps the same proportions (stratified). Class weights compensate for classes with few images.

**Models.** By default each task fine-tunes an ImageNet-pretrained EfficientNetB0. First only a new classification head is trained with the backbone frozen (learning rate 1e-3), then the top 40 layers of the backbone are unfrozen and fine-tuned at 1e-5. The project's original from-scratch CNN is still available (`--backbone custom`): two blocks of two 3×3 convolutions (5×5 for damage level and damage type) with ReLU, max pooling and batch normalization, then a 256-filter convolution, global average pooling, a 256-unit dense layer and 0.5 dropout. The original version flattened the last feature map into a 512-unit dense layer, which alone was about 205M parameters per task and was why running all 8 models was so demanding; global pooling brings the custom model under 1M parameters.

**Augmentation.** Random horizontal and vertical flips, rotation up to 40°, shifts up to 20% and zoom up to 30%, built into the model so they only run during training.

**Training.** Adam optimizer (a separate one per model), sparse categorical cross-entropy loss with a softmax output. Early stopping ends training once validation loss stops improving, the learning rate is reduced on plateaus, and the weights from the epoch with the lowest validation loss are the ones that get saved.

**Evaluation.** Each model is scored on its task's test set with accuracy, per-class precision/recall/F1, macro F1 and a confusion matrix.

# Usage
```bash
pip install -r requirements.txt
```

Put the Φ-Net arrays in one folder, either as `data/task<N>/task<N>_{X,y}_{train,test}.npy` or all directly in `data/`. Images must be `(N, 224, 224, 3)` RGB (0-255 or 0-1); labels can be one-hot or class ids.

```bash
# Train all 8 tasks (or pick some with --tasks 1 2 3)
python train.py --data-dir data --out-dir runs/effnet
# Original from-scratch CNN instead of the pretrained backbone
python train.py --data-dir data --out-dir runs/custom --backbone custom

# Test-set metrics, confusion matrices and a results table (runs/effnet/results.md)
python evaluate.py --data-dir data --runs-dir runs/effnet

# Classify new photos (any size; resized to 224x224)
python predict.py --runs-dir runs/effnet photo.jpg
```

On Google Colab, mount Drive, clone the repo and run the same commands with `!`, pointing `--data-dir` at the dataset folder on Drive. Use a GPU runtime and add `--mixed-precision` to train faster.

Each task's folder under the runs directory holds `model.keras`, `history.json`, `curves.png` (accuracy and loss curves), `report.txt` and `confusion_matrix.png`. Task names, class names (in label order) and default hyperparameters live in `config.py`. Check the class order there against the Φ-Net label definitions for your copy of the data.

| File | Purpose |
|---|---|
| `config.py` | Task definitions and hyperparameters |
| `data.py` | Loading, validation checks, stratified split, `tf.data` pipeline |
| `model.py` | Pretrained and custom model builders, augmentation |
| `train.py` | Trains and saves one model per task |
| `evaluate.py` | Test-set metrics and plots |
| `predict.py` | Predictions for new images |

# Results
Test-set results for the current pipeline have not been recorded yet. After training, run `evaluate.py` and paste `runs/<name>/results.md` here.

# The Dataset
Both AI and machine learning (ML) technologies have developing rapidly in recent decades, especially in the application of deep learning (DL) in computer vision (CV). The objective of ML and DL implementation is to have computers perform labor-intensive repetitive tasks while simultaneously “learning” from those tasks. Both ML and DL fall within the scope of empirical study, where data is the most essential component. In vision-based Structural Health Monitoring (SHM), using images as data media is currently an active research direction. Structural images obtained from reconnaissance efforts or daily life are playing an increasing role as the success of ML and DL is contingent on the volume of data media available. The expectation is that eventually computers will be able to realize autonomous recognition of structural damage in daily life—under service conditions—or after an extreme event—a large earthquake or extreme wind. Until now, vision-based SHM applications have not fully benefited from the data-driven CV technologies, even as interest on this topic is ever increasing. Its application to structural engineering has been hamstrung mainly due to two factors: (1) the lack of a general automated detection principles or frameworks based on domain knowledge; and (2) the lack of benchmark datasets with well-labeled large amounts of data [5].

To address the above mentioned drawbacks, there was a recent effort to build a large-scale open sourced structural image database: the PEER (Pacific Earthquake Engineering Research Center) Hub ImageNet (PHI-Net) as of November 2019 this Phi-Net Dataset contains 36,413 images with multiple attributes for the following baseline recognition tasks: scene level classification, structural component type identification, crack existence check and damage level detection. The Phi-Net dataset uses a hierarchy-tree framework for automated structural detection tasks founded on past experiences from reconnaissance efforts for post-earthquakes and other hazards. Through a tree-branch mechanism, each structural image can be clustered into several subcategories representing detection tasks. This acts as a sort of filtering operation to decrease the complexity of the problem and improve the performance of the automated applications of the algorithms. To the best of the authors’ knowledge, until now there was no open sourced structural image dataset with multi-attribute labels and this volume of images in the vision-based structural health monitoring area. It is believed that this image dataset and its corresponding detection tasks and framework will provide the necessary benchmark for future studies of deep learning in vision based structural health monitoring.

Analogous to the classification and localization tasks in the ImageNet challenge, the goal of the Φ-Net framework was to construct similar recognition tasks, designed for structural damage recognition and evaluation. Based on past experiences from reconnaissance efforts (Sezen et al. [2003]; Li and Mosalam [2013]; Mosalam et al. [2014]; and Koch et al. [2015]), several issues affect the safety of structures post-event: the type of damaged component, the severity of damage in the component, and the type of damage. Because images collected from reconnaissance efforts broadly vary, including, different distances from objects, camera angles, and emphasized targets, it is useful to cluster these issues into different levels. That is, images taken from a very close distance or only containing part of the component belong to the pixel level; major targets in images such as single or multiple components belong to the object level, and images containing most of the structure belong to the structural level. Moreover, the corresponding evaluation criteria will be different for different levels: that is, images in the pixel level are more related to the material type and damage status; images on the structural level are more related to the structural type and failure status [5].
Herein is a new processing framework with a hierarchy-tree structure shown in Figure 1, where images are classified as follows: (1) a raw image is clustered to different scene levels; (2) according to its level, corresponding recognition tasks are applied layer by layer following this hierarchy structure; and (3) each node is seen as one recognition task or a classifier, and the output of each node is seen as a characteristic or feature of the image to help with further analysis and decision making if required [5].

In the current Φ-Net, we designed the following eight benchmark classification tasks:

1.	three-class classification for scene level;
2.	binary classification for damage state;
3.	binary classification for spalling condition (material loss);
4.	binary classification for material type;
5.	three-class classification for collapse mode;
6.	four-class classification for component type;
7.	four-class classification for damage level and
8.	four-class classification for damage type.

The proposed framework with a hierarchy-tree structure is depicted in Figure 1, where grey boxes represent the detection tasks for the corresponding attributes, white boxes within the dashed lines are possible labels of the attributes to be chosen in each task, and ellipsis in boxes represent other choices or conditions. In the detection procedure, one starts from a root leaf where recognition tasks are conducted layer by layer and node by node (grey box) until another leaf node. The output label of each node describes a structural attribute. Each structural image may have multiple attributes, i.e., one image can be categorized as being at the pixel level, concrete, damaged state, etc. In the terminology of CV, this is considered a multi-attribute (multi-label) classification problem. Given that this is a pilot study, these attributes are treated independently at this stage. More extensions such as the multi-label version of Φ-Net will be updated in future studies.

The dataset exists at: https://apps.peer.berkeley.edu/phi-net/
