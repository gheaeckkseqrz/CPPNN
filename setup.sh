# Install python module parse, it's used to automaticaly generate C++ code to call OpenCL Kernels
cd library/Kernels
virtualenv .env
source .env/bin/activate
pip install parse
deactivate

# Back to repo root
cd -

# Generate the test data (require a working torch install)
# First need to set the VGG model path in generateTorchVGGTestData.lua
cd tests
echo "Downloading VGG Model"
wget -c https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/bb2b4fe0a9bb0669211cf3d0bc949dfdda173e9e/VGG_ILSVRC_19_layers_deploy.prototxt
wget -c http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel
echo "Generating Test Data - This may take quite a while"
mkdir -p TestData ; cd TestData
th ../generateTorchTestData.lua

cd ../.. #Back to root

# Running build and tests
mkdir -p build ; cd build
cmake .. && make && ./TestNN

