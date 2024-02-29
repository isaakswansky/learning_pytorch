# Intro

## Machine Learning (ML)

Deep Learning (DL) is a subset of Machine Learning (ML)

Machine Learning is a subset of Artificial Intelligence (AI)

### Traditional Programming

- Input
- use existing Rules
- to produce Output

### Machine Learning Algorithm

- Input
- Desired Output
- figure out Rules

### Why ML?

- We maybe don't know all the rules
- there are too many rules to consider

But:

- If you can build a simple rule-based system that doesn't require ML, do that.

### What DL is good for

- Problems with a long lists of rules
- Continually changing environments - need to adapt to new scenarios
- discovering insights wihtin large collections of data - too cumbersome to do it manually

### What DL is (typically) not good for

- When you meed eyplainability - DL models are typically uninterpretable by a human
- When the traditional approach is a better option
- When errors are unacceptable - outputs of DL are based on probability
- When you don't have much data (typically)

### ML vs. DL

#### ML

- input:
  - structured data (tables)
- algorithms:
  - gradient boosted machine
  - random forest
  - naive bayes
  - nearest neighbout
  - support vector machine
  - ...

#### DL

- input:
  - unstructured data (e.g. hand writing, images, audio)
- algorithms:
  - neural network
  - fully connected neural network
  - convolutional neural network
  - recurrent neural network
  - transformer
  - ...

## Neural Networks

input (unstructured) -> numerical encoding -> NN (input layer, hidden layers, output layer) -> representation outputs -> human understandable outputs (e.g. classification)

### Anatomy

Input Layer:

- data goes in here

Hidden Layers:

- learns patterns in data

Output Layer:

- outputs learned representation or prediction probabilities

## Types of Learning

### Supervised Learning

- a lot of data, a lot of examples, tagged/labeled data

### Unsupervised & Self-supervised Learning

- only data, no labels
- learns patterns, identifies clusters without knowing the classes

### Transfer Learning

- take patterns from one model and transfer it to another model

### Reinforcement Learning

- agent acts in an environment
- rewards actions

## What is DL used for?

- recommendation (e.g. YouTube, Spotify ...)
- Sequence to Sequence (seq2seq):
  - translation
  - speech recognition
- Classification/Regression:
  - computer vision (e.g. object detection)
  - natural language processing (e.g. spam detection)

## PyTorch

### What is PyTorch?

- popular DL framework
- write fast DL code in python (with GPU support)
- access pre-built DL models
- ecosystem for the whole stacK: preprocessing, model data, deployment
- originated from facebook, now open source

### Why Pytorch?

- most widely used DL framework

## What is a tensor?

- fundamental building block of neural networks
- numerical representation of data/information
- input and output of neural networks

## What are we going to cover?

- PyTorch basics & Fundamentals
- Preprocessing data (data into tensors)
- building and using pretrained DL models
- fitting model to data
- making predicitions with a model
- evaluatin model predictions
- saving and loading modles
- using a trained model to make predictions on custom data

## Pytorch Workflow

- Get data ready
- build or pick a pretrained model
  - pick loss function & optimizer
  - build training loop
- fit the model to the data and make a prediction
- evaluate the model
- improve through experimentation
- save and reload your trained model

