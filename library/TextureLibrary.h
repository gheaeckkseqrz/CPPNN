#ifndef __TEXTURE_LIBRARY_H__
#define __TEXTURE_LIBRARY_H__

#include <map>
#include <string>

#include "ParametricModel.h"
#include "Sequential.h"
#include "Tensor.h"

namespace NN
{
  class TextureLibrary
  {
  public:
    TextureLibrary();
    ~TextureLibrary();

    void addImage(std::string const &path);
    void addDirectory(std::string const &path);
    ParametricModel computeParametricModel(std::shared_ptr<Tensor> image);
    std::vector<std::string> findNN(std::shared_ptr<Tensor> example, int n, int layers = FULL_MODEL);

    void saveToFile(std::string const &path) const;
    void loadFromFile(std::string const &path);

  protected:
    std::vector<int> findBestIndices(std::vector<float> &data, int n);
    void pushModelToGPU(std::shared_ptr<Tensor> gpuBuffer, ParametricModel const &m, int offset, int layers);

  private:
    std::shared_ptr<Tensor> _models;
    std::shared_ptr<Sequential> _descriptorNetwork;
    std::map<std::string, ParametricModel> _library;

    int _maxProcessingBlockSize;
    int _maxLibraryCapacity;
  };
}

#endif /* __TEXTURE_LIBRARY_H__ */
