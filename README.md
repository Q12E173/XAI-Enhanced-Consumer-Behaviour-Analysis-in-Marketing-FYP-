# XAI-Enhanced Consumer Behaviour Analysis in Marketing

An AI-based NLP framework for consumer sentiment analysis using
deep learning, Transformer models, ensemble learning, and Explainable AI (XAI).

## Overview

Understanding customer feedback is important for identifying consumer
preferences, product experiences, and market trends. However, traditional
analysis methods can be difficult to scale when dealing with large volumes
of unstructured customer reviews.

This project develops an AI-based framework to analyse consumer sentiment
from Amazon product reviews. The framework combines deep learning and
Transformer-based models with Explainable AI (XAI) to improve both
prediction performance and model interpretability.

The project evaluates CNN, BERT, and RoBERTa models before applying
fine-tuning and a stacked ensemble approach. SHAP is then used to explain
the predictions of the final ensemble model.

---

## Project Objectives

The project aims to:

- Develop an AI framework to automate consumer behaviour analysis.
- Apply deep learning models to improve sentiment classification.
- Improve model performance through Transformer-based architectures
  and fine-tuning.
- Combine complementary Transformer models using ensemble learning.
- Apply Explainable AI techniques to improve model interpretability.
- Identify important textual features contributing to sentiment predictions.

---

## Dataset

The project uses the:

**Amazon US Customer Reviews Dataset**

The study focuses specifically on the:

**Personal Care Appliances** category.

The selected category contains approximately **85,982 reviews** and
includes product and review-related attributes such as product title,
review headline, review body, star rating, and other metadata.

The dataset was used for consumer sentiment analysis and NLP-based
consumer behaviour analysis.

> The original dataset is not included in this repository due to
> dataset size and distribution considerations.

---

## Methodology

The project consists of three main stages:

### 1. Initial Multi-Class Classification

Three models were initially developed:

- CNN
- BERT
- RoBERTa

The models classify reviews into three sentiment classes:

```text
Negative
Neutral
Positive
````

The initial experiment was used to establish baseline performance and
compare traditional deep learning with Transformer-based architectures.

---

### 2. Fine-Tuning

Due to the difficulty of distinguishing the Neutral class and the
imbalance between sentiment classes, the classification task was
simplified into two classes:

```text
Negative
Positive
```

BERT and RoBERTa were then fine-tuned using different configurations,
including different data splits, dropout settings, learning rates,
and regularisation strategies.

The best-performing configurations were selected for the final ensemble.

---

### 3. Stacked Ensemble

The best BERT and RoBERTa models were combined using a stacked ensemble.

The architecture is:

```text
                   Input Review
                       │
             ┌─────────┴─────────┐
             │                   │
          BERT 1             RoBERTa 2
             │                   │
       Softmax Prob.        Softmax Prob.
             │                   │
             └─────────┬─────────┘
                       │
              Concatenated Features
                       │
             Logistic Regression
               Meta-Classifier
                       │
                       ▼
             Final Sentiment
              Prediction
```

The two Transformer models generate probability distributions for the
Positive and Negative classes. These probabilities are concatenated
into a four-dimensional feature vector and passed to a Logistic
Regression meta-classifier.

---

## Model Performance

The experimental results showed a clear improvement from the CNN
baseline to Transformer-based models and finally to the stacked ensemble.

| Model                | Classification |          Accuracy |
| -------------------- | -------------- | ----------------: |
| CNN                  | 3-label        |               71% |
| BERT                 | 3-label        |               88% |
| RoBERTa              | 3-label        |               90% |
| BERT 1               | 2-label        |               94% |
| BERT 2               | 2-label        |               93% |
| RoBERTa 1            | 2-label        |            93.96% |
| RoBERTa 2            | 2-label        |            94.39% |
| **Stacked Ensemble** | **2-label**    | **94.70% (~95%)** |

The results demonstrate that Transformer-based models substantially
outperformed the CNN baseline. Combining BERT and RoBERTa through
stacked generalisation further improved the final prediction performance.

---

## BERT Fine-Tuning

The fine-tuned BERT models use:

* `bert-base-uncased`
* Binary classification
* AdamW optimizer
* Learning rate: `5e-6`
* Batch size: `16`
* 3 epochs
* Dropout: `0.3`
* Early stopping
* Weighted sampling for class imbalance

Two data split configurations were evaluated:

```text
BERT 1: 80% Train / 10% Validation / 10% Test
BERT 2: 70% Train / 15% Validation / 15% Test
```

BERT 1 achieved the better overall performance with an accuracy of 94%.

---

## RoBERTa Fine-Tuning

The fine-tuned RoBERTa models use:

* `roberta-base`
* Binary classification
* Maximum sequence length: 96
* Learning rate: `5e-6`
* Batch size: `16`
* 5 epochs
* Dropout and regularisation
* Weighted sampling for class imbalance

Two configurations were evaluated:

```text
RoBERTa 1: 80% Train / 10% Validation / 10% Test
RoBERTa 2: 70% Train / 15% Validation / 15% Test
```

RoBERTa 2 achieved the best overall balance between the Positive and
Negative classes and was selected for the final ensemble.

---

## Handling Class Imbalance

The Amazon reviews dataset contains substantially more Positive reviews
than Neutral and Negative reviews.

Different strategies were used depending on the model:

* Weighted Cross-Entropy Loss
* Class-weighted Focal Loss
* Weighted Random Sampling
* Label smoothing
* L2 regularisation
* Stratified data splitting

For the final ensemble, WeightedRandomSampler was used to provide more
balanced exposure to the minority class during training.

---

## Explainable AI with SHAP

To improve the interpretability of the final model, SHAP
(Shapley Additive Explanations) was applied to the stacked ensemble.

The ensemble was treated as a black-box predictor and SHAP was used
to identify how individual words or tokens contributed to sentiment
predictions.

### Global Explanations

The project generates:

* SHAP summary bar plots
* SHAP beeswarm plots
* SHAP token impact heatmaps

These visualisations identify the most influential words across the
review dataset.

### Local Explanations

The project also generates:

* SHAP force plots
* SHAP waterfall plots

These explain individual predictions by showing which words increased
or decreased the model's confidence toward a particular sentiment class.

---

## Example SHAP Insight

For an individual review, SHAP can show how specific words contribute
toward a Negative or Positive prediction.

For example:

```text
Input Review
     │
     ▼
Transformer Ensemble
     │
     ▼
Sentiment Prediction
     │
     ▼
SHAP Explanation
     │
     ├── Important words
     ├── Positive contribution
     └── Negative contribution
```

This allows the model prediction to be interpreted beyond simply
returning a sentiment label.

---

## Technologies

### Programming

* Python

### Machine Learning / Deep Learning

* PyTorch
* TensorFlow
* Scikit-learn

### NLP

* Hugging Face Transformers
* BERT
* RoBERTa
* Tokenization
* Text preprocessing

### Explainable AI

* SHAP

### Data Processing

* Pandas
* NumPy

### Visualisation

* Matplotlib
* Seaborn

### Development Environment

* Kaggle Notebook
* GPU acceleration

---

## How to Run

The notebooks were developed and executed in a GPU-enabled environment,
primarily using Kaggle Notebook.

The original Amazon US Customer Reviews Dataset is not included in this
repository. The dataset must be obtained separately before running the
notebooks.

Before running the notebooks:

1. Obtain the Amazon US Customer Reviews Dataset.
2. Place the dataset in the appropriate directory or update the dataset
   path in the notebook.
3. Install the required Python dependencies listed in `requirements.txt`.
4. Run the notebooks in the following order:

   1. CNN 3 Labels
   2. BERT 3 Labels
   3. RoBERTa 3 Labels
   4. BERT fine-tuning 
   5. RoBERTa fine-tuning
   6. Stacked ensemble and SHAP analysis

---

## Key Findings

The experiments demonstrated several important findings:

1. Transformer-based models substantially outperformed the CNN baseline.
2. RoBERTa achieved the strongest initial performance in the 3-label setup.
3. Removing the ambiguous Neutral class improved binary sentiment classification.
4. Fine-tuning improved BERT and RoBERTa performance.
5. BERT 1 and RoBERTa 2 were selected as the best models for ensemble construction.
6. The stacked ensemble achieved approximately 95% test accuracy.
7. SHAP provided global and local explanations for the ensemble predictions.
8. The combination of strong predictive performance and model
   interpretability provides a more transparent approach to consumer
   sentiment analysis.

---

## Limitations

The project was developed under computational resource constraints.
RoBERTa-large was therefore used only in the initial multi-class
experiment, while RoBERTa-base was used for the binary fine-tuning
experiments.

The dataset also contains class imbalance, particularly a much larger
number of Positive reviews.

The current framework focuses primarily on sentiment classification
rather than directly predicting purchasing behaviour.

---

## Future Improvements

Potential future improvements include:

* Training larger Transformer architectures.
* Using larger GPU or cloud computing resources.
* Expanding the analysis to additional Amazon product categories.
* Incorporating more detailed emotion categories.
* Developing a real-time sentiment analysis application.
* Integrating additional XAI techniques.
* Exploring more advanced ensemble architectures.
* Connecting sentiment insights with product and marketing analytics.

---

## Academic Project

**Project Title:** XAI-Enhanced Consumer Behaviour Analysis in Marketing

**Programme:** Bachelor of Computer Science (Hons) Artificial Intelligence

**Institution:** Multimedia University

**Year:** 2025

**Author:** Ng Le Qian
