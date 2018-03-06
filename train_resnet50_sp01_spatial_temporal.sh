#!/usr/bin/env sh

TOOLS=/home/junchao/workspace/caffe-rc3-lstm/build/tools

$TOOLS/caffe train --solver=configure/resnet50_cvgj_spatial_temporal_solver_sp01.prototxt --weights="path to caffemodel" --gpu=0
