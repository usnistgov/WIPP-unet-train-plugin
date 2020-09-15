# UNet CNN Semantic-Segmentation Training plugin

## Docker distribution

This plugin is available on DockerHub from the WIPP organization

```
docker pull wipp/wipp-unet-cnn-train-plugin
```

## Build Docker File
```bash
#!/bin/bash

version=0.0.1
docker build . -t wipp/wipp-unet-cnn-train-plugin:latest
docker build . -t wipp/wipp-unet-cnn-train-plugin:${version}
```

## Run Docker File

```bash
docker run --gpus device=all \
    -v "path/to/input/data/images":/data/images \
    -v "path/to/input/data/masks":/data/masks \
    -v "path/to/output/folder":/data/outputs \
    wipp/wipp-unet-cnn-train-plugin \
    --batchSize 8 \
    --outputDir /data/outputs \
    --tensorboardDir /data/outputs \
    --imageDir /data/images \
    --maskDir /data/masks \
    --testEveryNSteps 1000
```

## UNet Training Job Options
```bash
usage: train_unet [-h] --imageDir IMAGE_DIR --maskDir MASK_DIR
                  [--useTiling USE_TILING] [--tileSize TILE_SIZE]
                  [--trainFraction TRAIN_FRACTION] [--batchSize BATCH_SIZE]
                  [--numberClasses NUMBER_CLASSES]
                  [--learningRate LEARNING_RATE] --outputDir OUTPUT_DIR
                  --tensorboardDir TENSORBOARD_DIR
                  [--testEveryNSteps TEST_EVERY_N_STEPS]
                  [--balanceClasses BALANCE_CLASSES]
                  [--useIntensityScaling USE_INTENSITY_SCALING] 
                  [--useAugmentation USE_AUGMENTATION]
                  [--augmentationReflection AUGMENTATION_REFLECTION]
                  [--augmentationRotation AUGMENTATION_ROTATION]
                  [--augmentationJitter AUGMENTATION_JITTER]
                  [--augmentationNoise AUGMENTATION_NOISE]
                  [--augmentationScale AUGMENTATION_SCALE]
                  [--augmentationBlurMaxSigma AUGMENTATION_BLUR_MAX_SIGMA]
                  [--augmentationIntensity AUGMENTATION_INTENSITY]

```