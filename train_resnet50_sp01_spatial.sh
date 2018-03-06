#!/usr/bin/env sh

TOOLS=caffe-rc3-lstm/build/tools

$TOOLS/caffe train --solver=configure/resnet50_cvgj_spatial_solver_sp01.prototxt --weights=PretrainedModel/resnet50_cvgj_iter_320000.caffemodel --gpu=0
