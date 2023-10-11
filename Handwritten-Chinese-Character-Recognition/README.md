# Handwritten Chinese Character Recognition with Neural Networks
This is the final project for CSC413H1 Winter 2023 (Neural Networks and Deep Learning), taught by Jimmy Ba and Bo Wang at the University of Toronto.<br />
Project Handout can be found here https://uoft-csc413.github.io/2023/assets/misc/project_handout.pdf

## Abstract
In this project, we explore the effectiveness of transfer learning using two pre-trained models, EfficientNet and InceptionV3+Gated Recurrent Unit (GRU), for Handwritten Chinese Character Recognition (HCCR) on the Chinese MNIST dataset. We fine-tune these models on various training datasets and evaluate their performance. Our results demonstrate that augmenting the size and diversity of the training dataset can enhance model performance. These findings suggest that pre-trained models are a promising approach for HCCR and can be further improved by utilizing larger and more diverse training datasets.

## Data
For training and evaluating our models, we utilized the Chinese MNIST dataset, which was created by Gabriel Preda and is available on [Kaggle](https://www.kaggle.com/datasets/gpreda/chinese-mnist). To explore the association between dataset size and model performance, as well as to address the computational challenges posed by the complex InceptionV3 model architecture and GRU layer, we randomly generated three subsets of the full dataset for training in each experiment.

## Models
### EfficientNet
<img width="468" alt="EfficientNetV2" src="https://github.com/dabandaidai/CSC413/assets/105243552/148355a3-da6e-43bb-8841-3ec9b7ec27d1">

To run EfficientNet Model, execute `EfficientNet\src.ipynb` with corresponding datasets in `EfficientNet\`.

### InceptionV3+GRU
![inceptionv3onc--oview_vjAbOfw](https://github.com/dabandaidai/CSC413/assets/105243552/94e11319-177c-46cc-8572-becba5887d93)
The output sequence of InceptionV3 shown above is then passed into a GRU layer.

To run InceptionV3+GRU Model, execute `InceptionV3+GRU\InceptionV3 + GRU.ipynb` with dataset `.\ChineseMNIST`.

## Report
See `CSC413_report.pdf` for detailed experiments, results and analysis.
