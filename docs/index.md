# Spatio-Temporal Attention-based Unet for Field Boundary Detection

This is the first place solution of team `WeMoveMountains` in the NASA Harvest Field Boundary Detection Challenge. This solution is a single 10-fold modified Regnetv-Unet developed in Pytorch.

![{{model_nasa_rwanda_field_boundary_competition_gold_v1}}](https://zindi-public-release.s3.eu-west-2.amazonaws.com/uploads/competition/image/331/header_21ba8a01-ef4a-43c4-af10-8c5bab32d572.png)

MLHub model id: `model_nasa_rwanda_field_boundary_competition_gold_v1`. Browse on [Radiant MLHub](https://mlhub.earth/model/model_nasa_rwanda_field_boundary_competition_gold_v1).

## Training Data

- [Training Data Source](https://api.radiant.earth/mlhub/v1/collections/nasa_rwanda_field_boundary_competition_source_train)
- [Training Data Labels](https://api.radiant.earth/mlhub/v1/collections/nasa_rwanda_field_boundary_competition_labels_train)


## Related MLHub Dataset

The dataset description is available on RadiantMLHub. See [here](https://mlhub.earth/data/nasa_rwanda_field_boundary_competition).


## Citation

Muhamed T., Azer K. (2023) “Spatio-Temporal Attention-based Unet for Field Boundary Detection”, Version 1.0, Radiant MLHub. [Date Accessed]
Radiant MLHub. <https://doi.org/10.34911/rdnt.u7n266>

## License

 [CC-BY-4.0](../LICENSE)

## Creators

This solution was developped by:
* [Muhamed Tuo](https://www.linkedin.com/in/muhamed-tuo-b1b3a0162/)
* [Azer Ksouri](https://github.com/ASSAZZIN-01)

## Learning Approach

- Supervised Learning

## Prediction Type

- Segmentation

## Model Architecture

Our solution is a modified Unet++ with an attention mechanism between every interconnection of the encoder to the decoder (Meaning after each output layer of the encoder model). The encoder is a Regnet (more specifically `regnetv_040` available in the [Timm library](https://github.com/huggingface/pytorch-image-models)). 

## Training Operating System

The training was done on a Linux system with an Nvidia GPU (A100 80GB). An A600 24GB should be enough to train the model.

## Model Inferencing

Review the [GitHub repository README](../README.md) to get started running
this model for new inferencing.

## Training

* Augmentation

The only augmentation we did was static. It is done before the training and saved into a new folder. We noticed that the model learned better with little (only Flip augmentation) to no augmentation during the training.

* Training procedure

For each tile, all the time-series images are loading as `Timestamps x C x H x W` and passed to the model. The input now is `Batch_size x TimeStamps x C x H x W`. It is reshaped to `Batch_size*TimeStamps x C x H x W` before being fed into the encoder, and then reshaped to `Batch_size x TimeStamps x D x H' x W'` for the attention pooling mechanism. Finally, the input to the decoder becomes `Batch_size x D" x H" x W"`.

### Structure of Output Data

The output file is a `csv` named `output.csv` and should be available in the `data/output` folder. Each row of the csv file correspond to a pixel of the flattened 256x256 image.
