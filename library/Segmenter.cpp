#include "Kmeans.h"
#include "OpenCLFuncs.h"
#include "Segmenter.h"
#include "TorchLoader.h"

namespace NN
{
  Segmenter::Segmenter(std::string const &networkFile)
  {
    _network = std::dynamic_pointer_cast<Sequential>(TorchLoader::getInstance()->loadFile(networkFile));
    _layers = std::vector<int>({2, 8, 14, 26, 38});
    _weights = std::vector<int>({1, 1, 1, 4, 1});
    std::vector<bool> retainPolicy(_network->size(), false);
    for (int l : _layers)
      retainPolicy[l] = true;
    _network->setRetainPolicy(retainPolicy);
  }

  std::shared_ptr<Tensor> Segmenter::createIndexesMask(std::shared_ptr<Tensor> input) const
  {
    std::shared_ptr<Tensor> maps = extractFeatureMaps(input);
    Kmeans clusterer(2);
    std::shared_ptr<Tensor> indexesMask = clusterer.clusterData(maps);
    return indexesMask;
  }

  std::shared_ptr<Tensor> Segmenter::createRGBMask(std::shared_ptr<Tensor> input) const
  {
    return indexesMaskToRGB(createIndexesMask(input));
  }

  std::shared_ptr<Tensor> Segmenter::indexesMaskToRGB(std::shared_ptr<Tensor> input) const
  {
    std::vector<float> colorsData = std::vector<float>({230, 25, 75,
          60, 180, 75,
          255, 225, 25,
          0, 130, 200,
          245, 130, 48,
          145, 30, 180,
          70, 240, 240,
          240, 50, 230,
          210, 245, 60,
          250, 190, 190,
          0, 128, 128,
          230, 190, 255,
          170, 110, 40,
          255, 250, 200,
          128, 0, 0,
          170, 255, 195,
          128, 128, 0,
          255, 215, 180,
          0, 0, 128,
          128, 128, 128,
          255, 255, 255,
          0, 0, 0});
    std::shared_ptr<Tensor> colors = std::make_shared<Tensor>(std::vector<int>({22, 3}), colorsData);
    std::vector<int> outputSizes({3, input->getSize(0), input->getSize(1)});
    std::shared_ptr<Tensor> output = std::make_shared<Tensor>(outputSizes);
    OpenCLFuncs::getInstance()->indexesToRGB(*input, *colors, *output, output->getNbElements());
    return output;
  }

  std::shared_ptr<Tensor> Segmenter::extractFeatureMaps(std::shared_ptr<Tensor> input) const
  {
    _network->forward(input);
    int totalChannels = 0;
    for (auto layer : _layers)
      totalChannels += _network->get(layer)->getOutput()->getSize(0);
    std::vector<int> sizes({totalChannels, input->getSize(1), input->getSize(2)});
    std::shared_ptr<Tensor> maps = std::make_shared<Tensor>(sizes);
    std::cout << maps->means() << std::endl;
    int currentOffset = 0;
    for (int i(0) ; i < _layers.size() ; ++i)
      {
        std::shared_ptr<Tensor> output = _network->get(_layers[i])->getOutput();
	output->mul(_weights[i]);
        (*maps)[std::pair<int, int>(currentOffset, currentOffset + output->getSize(0))].copy(*output);
      }
    return maps;
  }
}
