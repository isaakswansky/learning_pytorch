# Classification

## Binary Classification

one thing or another (spam / no spam)

## Multiclass Classification

more than one thing or another

## Multilabel Classification

multiple label options per sample

## Classification Inputs and Outputs

Inputs -> ML Algorithm -> Outputs

### Inputs

- e.g. RGB Images (224, 224, 3)
- shape: [batch_size, colour_channels, width, height]

### Outputs

- prediction value (0.0 to 1.0) for each class
- transfer to labels (e.g. highest prediction value)
- shape = [number of classes]

## Architecture of a classification model

