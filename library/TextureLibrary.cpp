#include <experimental/filesystem>
#include <iostream>
#include <fstream>
#include <numeric>

#include "../MatToTensor.h"
#include "OpenCLFuncs.h"
#include "TextureLibrary.h"
#include "TorchLoader.h"

namespace NN
{
  TextureLibrary::TextureLibrary()
  {
    _maxLibraryCapacity = 5;
    _maxProcessingBlockSize = 256;
    _descriptorNetwork = std::dynamic_pointer_cast<Sequential>(TorchLoader::getInstance()->loadFile("../tests/TestData/vgg.t7"));
  }

  TextureLibrary::~TextureLibrary()
  {
    // std::cout << "    ~TextureLibrary();" << std::endl;
  }

  void TextureLibrary::addImage(std::string const &path)
  {
    std::shared_ptr<Tensor> t = tensorFromImage(path, 512, 15);
    if (t != nullptr)
      {
	ParametricModel model = computeParametricModel(t);
	_library[path] = model;
      }
  }

  void TextureLibrary::addDirectory(std::string const &path)
  {
    std::experimental::filesystem::recursive_directory_iterator d(path);
    int i = 0;
    for(auto &f : d)
      {
	if (_library.count(f.path().string()) == 0)
	  {
	    addImage(f.path().string());
	    std::cout << "[" << i++ << "]" << f << std::endl;
	    if (i > _maxLibraryCapacity)
	      break;
	  }
      }
    std::cout << "Library contains " << _library.size() << " entries" << std::endl;
  }

  ParametricModel TextureLibrary::computeParametricModel(std::shared_ptr<Tensor> image)
  {
    image->sub(127); // -- Preprocessing image for VGG
    _descriptorNetwork->forward(image);
    std::vector<float> relu1_1 = _descriptorNetwork->get(1)->getOutput()->covariance(true, true)->read();
    std::vector<float> relu2_1 = _descriptorNetwork->get(6)->getOutput()->covariance(true, true)->read();
    std::vector<float> relu3_1 = _descriptorNetwork->get(11)->getOutput()->covariance(true, true)->read();
    std::vector<float> relu4_1 = _descriptorNetwork->get(20)->getOutput()->covariance(true, true)->read();
    std::vector<float> relu5_1 = _descriptorNetwork->get(29)->getOutput()->covariance(true, true)->read();
    return ParametricModel(relu1_1, relu2_1, relu3_1, relu4_1, relu5_1);
  }

  std::vector<std::string> TextureLibrary::findNN(std::shared_ptr<Tensor> example, int n, int layers)
  {
    std::vector<std::pair<float, std::string>> bestResults;
    std::shared_ptr<Tensor> exampleFullModel = std::make_shared<Tensor>(computeParametricModel(example).getFullModel(layers));
    std::shared_ptr<Tensor> indexes = std::make_shared<Tensor>(std::vector<float>(_maxProcessingBlockSize, 1.0f));
    std::shared_ptr<Tensor> models = std::make_shared<Tensor>(std::vector<int>({_maxProcessingBlockSize, ParametricModel::getModelSize(layers)}));
    int i = 0;
    int totalTested = 0;
    std::vector<std::string> paths;
    for (std::pair<std::string, ParametricModel> m : _library)
      {
    	pushModelToGPU(models, m.second, i, layers);
	paths.push_back(m.first);
    	i++;
	totalTested++;
	if (i == _maxProcessingBlockSize || totalTested == _library.size())
	  {
	    Tensor t = models->transpose();
	    t.sub(*exampleFullModel);
	    t.pow(2.0f);
	    std::shared_ptr<Tensor> scratch = std::make_shared<Tensor>(std::vector<int>({t.getSize(1) + 1, t.getSize(0)}));
	    OpenCLFuncs::getInstance()->kmeansReductionInit(t.transpose(), *scratch, *indexes, 1.0f, scratch->getNbElements());

	    int offset = scratch->getSize(1);
	    int length = scratch->getSize(1);
	    while (offset > 1)
	      {
		offset = (length / 2) + (length % 2);
		OpenCLFuncs::getInstance()->kmeansReductionStep(*scratch, offset, length, scratch->getNbElements());
		length = offset;
	      }
	    std::shared_ptr<Tensor> res = std::make_shared<Tensor>(std::vector<int>({_maxProcessingBlockSize}));
	    OpenCLFuncs::getInstance()->kmeansRetreiveResults(*scratch, *res, res->getSize(0));

	    std::vector<float> distances = res->read();
	    std::vector<int> bestIndexes = findBestIndices(distances, n);
	    for (auto i : bestIndexes)
		bestResults.push_back(std::pair<float, std::string>(distances[i], paths[i]));

	    std::sort(std::begin(bestResults), std::end(bestResults), [](std::pair<float, std::string> a, std::pair<float, std::string> b) { return a.first < b.first; });
	    bestResults.resize(n);
	    i = 0;
	    paths.clear();
	  }
      }

    paths.clear();
    for (std::pair<float, std::string> r : bestResults)
      paths.push_back(r.second);
    return paths;
  }

  void TextureLibrary::saveToFile(std::string const &path) const
  {
    std::cout << "Saving to " << path << std::endl;
    std::ofstream myfile;
    myfile.open (path, std::ios::out | std::ios::binary);
    int nbEntries = _library.size();
    myfile.write((char*)&nbEntries, sizeof(int));
    for (std::pair<std::string, ParametricModel> m : _library)
      {
	int pathLength = m.first.size();
	myfile.write((char*)&pathLength, sizeof(int));
	myfile.write(m.first.c_str(), pathLength);
	myfile.write((char*)m.second.getFullModel().data(), ParametricModel::getModelSize() * sizeof(float));
      }
    std::cout << "Saved " << nbEntries << " to database [" << path << "]" << std::endl;
  }

  void TextureLibrary::loadFromFile(std::string const &path)
  {
    std::cout << "Loading from " << path << std::endl;
    std::ifstream myfile;
    myfile.open (path, std::ios::in | std::ios::binary);
    int nbEntries;
    myfile.read((char*)&nbEntries, sizeof(int));

    for (int i(0) ; i < nbEntries ; ++i)
      {
	int pathLength;
	myfile.read((char*)&pathLength, sizeof(int));
	std::vector<char> path(pathLength, 0);
	myfile.read(path.data(), pathLength);
	std::vector<float> model(ParametricModel::getModelSize(), 0);
	myfile.read((char*)model.data(), ParametricModel::getModelSize() * sizeof(float));
	_library[std::string(path.begin(), path.end())] = ParametricModel(model);
      }
    std::cout << "Loaded " << nbEntries << " from saved database [" << path << "]" << std::endl;
  }

  std::vector<int> TextureLibrary::findBestIndices(std::vector<float> &data, int n)
  {
    std::vector<int> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0); // fill with 0,1,2,...
    std::partial_sort(indices.begin(), indices.begin() + n, indices.end(),
                     [&data](int i,int j) {return data[i] < data[j];});
    return std::vector<int>(indices.begin(), indices.begin() + n);
  }

  void TextureLibrary::pushModelToGPU(std::shared_ptr<Tensor> gpuBuffer, ParametricModel const &m, int offset, int layers)
  {
    std::vector<float> fullModel = m.getFullModel(layers);
    (*gpuBuffer)[offset]->copy(fullModel);
  }
}
