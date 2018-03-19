# CPPNN

## SETUP

```sh
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
cd tests ; mkdir TestData ; cd TestData
th ../generateTorchTestData.lua
```
