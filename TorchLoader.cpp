#include <iostream>

#include "TorchConvolution.h"
#include "TorchLoader.h"
#include "TorchMaxPooling.h"
#include "TorchNumber.h"
#include "TorchRelu.h"
#include "TorchSequential.h"
#include "TorchStorage.h"
#include "TorchString.h"
#include "TorchTable.h"
#include "TorchTensor.h"

namespace NN
{
  TorchLoader *TorchLoader::_instance = nullptr;

  TorchLoader::TorchLoader()
  {
  }

  TorchLoader *TorchLoader::getInstance()
  {
    if (_instance == nullptr)
      _instance = new TorchLoader();
    return _instance;
  }

  std::shared_ptr<TorchObject> TorchLoader::loadFile(std::string const &torchFilePath)
  {
    std::map<int, std::shared_ptr<TorchObject>> loaded;
    std::ifstream file(torchFilePath);
    int type = std::atoi(readNextLine(file).c_str());
    return create(type, file, loaded);
  }

  std::shared_ptr<TorchObject> TorchLoader::create(int objectType, std::ifstream &file, std::map<int, std::shared_ptr<TorchObject>> &loaded)
  {
    switch (objectType)
      {
      case 0:
	break;
      case 1:
	return (new TorchNumber())->loadFromFile(file, loaded);
	break;
      case 2:
	return (new TorchString())->loadFromFile(file, loaded);
	break;
      case 3:
	return (new TorchTable())->loadFromFile(file, loaded);
	break;
      case 4:
	{
	int objectId = std::atoi(readNextLine(file).c_str());
	if (loaded[objectId] != nullptr)
	  return loaded[objectId];
	readNextLine(file); // VERSION
	readNextLine(file); // VERSION
	readNextLine(file); // String Len
	std::string className = readNextLine(file); // ClassName
	// std::cout << "Classname is " << className << std::endl;
	if (className == "Work in progress")
	  {
	  }
	// if (className == "nn.Dropout")
	// 	  loaded[objectId] = new Dropout();
	else if (className == "torch.FloatTensor" || className == "torch.LongTensor")
	  loaded[objectId] = (new TorchTensor())->loadFromFile(file, loaded);
	else if (className == "torch.FloatStorage" || className == "torch.LongStorage")
	  loaded[objectId] = (new TorchStorage())->loadFromFile(file, loaded);
	// else if (className == "nn.CAddTable")
	// 	loaded[objectId] = new CAddTable();
	// else if (className == "nn.ConcatTable")
	// 	loaded[objectId] = new ConcatTable();
	// else if (className == "nn.Identity")
	// 	loaded[objectId] = new Identity();
	// else if (className == "nn.Linear")
	// 	loaded[objectId] = new FullyConnected();
	// else if (className == "nn.MulConstant")
	// 	loaded[objectId] = new MulConstant();
	// else if (className == "nn.PixelShuffle")
	// 	loaded[objectId] = new PixelShuffle();
	// else if (className == "nn.SpatialReflectionPadding")
	// 	loaded[objectId] = new ReflectionPadding();
	else if (className == "nn.ReLU")
	  loaded[objectId] = (new TorchRelu())->loadFromFile(file, loaded);
	else if (className == "nn.Sequential")
	  loaded[objectId] = (new TorchSequential())->loadFromFile(file, loaded);
	// else if (className == "nn.ShaveImage")
	// 	loaded[objectId] = new ShaveImage();
	// else if (className == "nn.SoftMax")
	// 	loaded[objectId] = new Softmax();
	// else if (className == "nn.SpatialBatchNormalization")
	// 	loaded[objectId] = new BatchNormalisation();
	else if (className == "nn.SpatialMaxPooling")
	  loaded[objectId] = (new TorchMaxPooling())->loadFromFile(file, loaded);
	else if (className == "nn.SpatialConvolution")
	  loaded[objectId] = (new TorchConvolution())->loadFromFile(file, loaded);
	// else if (className == "nn.SpatialDilatedConvolution")
	// 	loaded[objectId] = new DilatedConvolution();
	// else if (className == "nn.SpatialFullConvolution")
	// 	loaded[objectId] = new FullConvolution();
	// else if (className == "nn.SpatialUpSamplingNearest")
	// 	loaded[objectId] = new SpatialUpsamplingNearest();
	// else if (className == "nn.Tanh")
	// 	loaded[objectId] = new Tanh();
	// else if (className == "nn.TotalVariation")
	// 	loaded[objectId] = new TotalVariation();
	// else if (className == "nn.View")
	// 	loaded[objectId] = new View();
	else
	  {
	    std::cerr << "Unknown class " <<  className << std::endl;
	    return nullptr;
	  }
	return loaded[objectId];
	}
	break;
      case 5:
	return (new TorchNumber())->loadFromFile(file, loaded);
	break;
      }
    return nullptr;
  }
}
