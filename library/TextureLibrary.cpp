#include <experimental/filesystem>
#include <iostream>

#include "../MatToTensor.h"
#include "TextureLibrary.h"
#include "TorchLoader.h"

namespace NN
{
  TextureLibrary::TextureLibrary()
  {
    _descriptorNetwork = std::dynamic_pointer_cast<Sequential>(TorchLoader::getInstance()->loadFile("../tests/TestData/vgg.t7"));
  }

  void TextureLibrary::addImage(std::string const &path)
  {
	std::shared_ptr<Tensor> t = tensorFromImage(path, 512);
	if (t != nullptr)
	  {
	    t->sub(127); // -- Preprocessing image for VGG
	    _library[path] = computeParametricModel(t);
	  }
  }

  void TextureLibrary::addDirectory(std::string const &path)
  {
    std::experimental::filesystem::recursive_directory_iterator d(path);
    for(auto &f : d)
      {
	addImage(f.path());
	std::cout << f << std::endl;
      }
    std::cout << "Library contains " << _library.size() << " entries" << std::endl;
  }

  ParametricModel TextureLibrary::computeParametricModel(std::shared_ptr<Tensor> image)
  {
    _descriptorNetwork->forward(image);
    std::vector<float> relu1_1 = _descriptorNetwork->get(1)->getOutput()->read();
    std::vector<float> relu2_1 = _descriptorNetwork->get(6)->getOutput()->read();
    std::vector<float> relu3_1 = _descriptorNetwork->get(11)->getOutput()->read();
    std::vector<float> relu4_1 = _descriptorNetwork->get(20)->getOutput()->read();
    std::vector<float> relu5_1 = _descriptorNetwork->get(29)->getOutput()->read();
    return ParametricModel(relu1_1, relu2_1, relu3_1, relu4_1, relu5_1);
  }
}
