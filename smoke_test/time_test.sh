export PYTHONPATH=$PYTHONPATH:../python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../external/mkldnn/install/lib

export KMP_AFFINITY=compact,1,0,granularity=fine

echo $@
../build/tools/caffe time  -model=./unet_prediction.prototxt -iterations=20  $@
