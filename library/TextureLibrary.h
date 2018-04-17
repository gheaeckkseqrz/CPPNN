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

    void addImage(std::string const &path);
    void addDirectory(std::string const &path);
    ParametricModel computeParametricModel(std::shared_ptr<Tensor> image);
    std::vector<std::string> findNN(std::shared_ptr<Tensor> example, int n);

  private:
    std::shared_ptr<Sequential> _descriptorNetwork;
    std::map<std::string, ParametricModel> _library;
  };
}

#endif /* __TEXTURE_LIBRARY_H__ */
