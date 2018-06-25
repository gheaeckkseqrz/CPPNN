#include "TorchLoader.h"
#include "TorchNumber.h"
#include "TorchSpatialUpsamplingNearest.h"
#include "TorchTable.h"

namespace NN
{
  TorchSpatialUpsamplingNearest::TorchSpatialUpsamplingNearest()
  {
  }

  std::shared_ptr<TorchObject> TorchSpatialUpsamplingNearest::loadFromFile(std::ifstream &file, std::map<int, std::shared_ptr<TorchObject>> &loaded)
  {
    int tableId = readNextInt(file);
    std::shared_ptr<TorchTable> table = std::dynamic_pointer_cast<TorchTable>(TorchLoader::getInstance()->create(tableId, file, loaded));
    _upscaleFactor = (int)std::dynamic_pointer_cast<TorchNumber>(table->get("scale_factor"))->value();
    return std::shared_ptr<TorchObject>(this);
  }
}
