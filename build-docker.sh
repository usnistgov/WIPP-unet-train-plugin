#!/bin/bash

docker build . -t wipp/wipp-unet-cnn-train-plugin:latest
docker build . -t wipp/wipp-unet-cnn-train-plugin:$1
