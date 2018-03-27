#ifndef __KMEANS_H__
#define __KMEANS_H__

#include "Tensor.h"

namespace NN
{
  class Kmeans
  {
  public:
    Kmeans(int nbCluster);

    std::shared_ptr<Tensor> clusterData(std::shared_ptr<Tensor> data, int maxIteration = 50);
    std::shared_ptr<Tensor> getCentroids() const;

  protected:
    void initCentroids(std::shared_ptr<Tensor> data);
    void updateCentroids(std::shared_ptr<Tensor> data);

  private:
    int _nbCluster;
    std::shared_ptr<Tensor> _centroids;
    std::shared_ptr<Tensor> _indexes;
};
}

#endif
