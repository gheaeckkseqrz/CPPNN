#include <algorithm>
#include "Kmeans.h"
#include "OpenCLFuncs.h"

namespace NN
{
  Kmeans::Kmeans(int nbCluster)
    :_nbCluster(nbCluster)
  {
  }

  std::shared_ptr<Tensor> Kmeans::clusterData(std::shared_ptr<Tensor> data, int maxIteration)
  {
    std::vector<int> backupSizes = data->getSizes();
    data->flatten();
    initCentroids(data);

    for (int i(0) ; i < maxIteration ; ++i)
      {
        OpenCLFuncs::getInstance()->kmeans(*data, *_centroids, *_indexes, 0, _indexes->getNbElements());
	updateCentroids(data);
      }
    data->setSizes(backupSizes);
    _indexes->setSizes(std::vector<int>(data->getSizes().begin() + 1, data->getSizes().end()));
    return _indexes;
  }

  std::shared_ptr<Tensor> Kmeans::getCentroids() const
  {
    return _centroids;
  }

  void Kmeans::initCentroids(std::shared_ptr<Tensor> data)
  {
    Tensor transposedData = data->transpose();
    _centroids = std::make_shared<Tensor>(std::vector<int>({_nbCluster, data->getSize(0)}));
    (*_centroids)[0].copy(data->means());
    _indexes = std::make_shared<Tensor>(std::vector<int>({data->getSize(1)}));
    for (int i(1) ; i < _nbCluster ; ++i)
      {
        OpenCLFuncs::getInstance()->kmeans(*data, (*_centroids)[std::pair<int, int>(0, i)], *_indexes, 2, _indexes->getNbElements());
        std::vector<float> res = _indexes->read();
        int max = std::distance(res.begin(), std::max_element(res.begin(), res.end()));
        (*_centroids)[i].copy(transposedData[max]);
      }
  }

  void Kmeans::updateCentroids(std::shared_ptr<Tensor> data)
  {
    std::vector<float> dataArray = data->read();
    std::vector<float> indexes = _indexes->read();
    std::vector<float> newCentroidsArray(_centroids->getNbElements());
    std::vector<float> counter(_centroids->getSize(0));

    int channelSize = data->getNbElements() / data->getSize(0);
    for (int i = 0 ; i < dataArray.size() ; ++i)
      {
        int channel = i / channelSize;
        int sample = i % channelSize;
        int index = (int)indexes[sample];
        newCentroidsArray[index * _centroids->getSize(1) + channel] += dataArray[i];
        if (channel == 0)
          counter[index]++;
      }
    _centroids->copy(newCentroidsArray);
    Tensor divTensor(counter);
    _centroids->div(divTensor);
  }

}
