#include "../MatToTensor.h"
#include "Module.h"
#include "TorchLoader.h"
#include "VGGPreprocessing.h"
#include "WCT.h"

std::shared_ptr<NN::Tensor> blendImage(std::shared_ptr<NN::Tensor> t1, std::shared_ptr<NN::Tensor> t2, float a)
{
  assert(t1->getSizes() == t2->getSizes());
  std::shared_ptr<NN::Tensor> res = std::make_shared<NN::Tensor>(t1->getSizes());
  res->copy(*t1);
  res->mul(a);
  std::shared_ptr<NN::Tensor> res2 = std::make_shared<NN::Tensor>(t2->getSizes());
  res2->copy(*t2);
  res2->mul(1 - a);
  res->add(*res2);
  return res;
}

std::shared_ptr<NN::Tensor> WCT(std::pair<std::shared_ptr<NN::Module>, std::shared_ptr<NN::Module>> codec, std::shared_ptr<NN::Tensor> content, std::shared_ptr<NN::Tensor> style)
{
  std::shared_ptr<NN::Tensor> contentFeatures = codec.first->forward(content);
  std::vector<int> featuresSize = contentFeatures->getSizes();
  std::shared_ptr<NN::Tensor> styleFeatures = codec.first->forward(style);
  styleFeatures->flatten();
  contentFeatures->flatten();

  NN::WCT transform;
  std::shared_ptr<NN::Tensor> transformedFeatures = transform.enforceCovariance(contentFeatures, styleFeatures->covariance(), std::make_shared<NN::Tensor>(styleFeatures->means()));
  transformedFeatures->setSizes(featuresSize);

  std::shared_ptr<NN::Tensor> result = codec.second->forward(transformedFeatures);
  return result;
}

int main(int ac, char ** av)
{
  std::cout << "VIDEO Style Transfer" << std::endl;
  if (ac < 3)
    {
      std::cerr << "Usage : " << av[0] << " CONTENT_IMAGES_SOURCE_FOLDER STYLE_IMAGE" << std::endl;
      return 1;
    }

  std::vector<std::pair<std::shared_ptr<NN::Module>, std::shared_ptr<NN::Module>>> networks;
  for (int i(5) ; i > 0 ; i--)
    {
      std::cout << "Loading encoder/decoder for relu " << i << std::endl;
      std::shared_ptr<NN::Module> vgg_encoder = std::dynamic_pointer_cast<NN::Module>(NN::TorchLoader::getInstance()->loadFile("models/vgg" + std::to_string(i) + "_encoder.t7"));
      std::shared_ptr<NN::Module> vgg_decoder = std::dynamic_pointer_cast<NN::Module>(NN::TorchLoader::getInstance()->loadFile("models/vgg" + std::to_string(i) + "_decoder.t7"));
      networks.push_back(std::make_pair(vgg_encoder, vgg_decoder));
    }
  NN::VGGPreprocessing p;
  std::shared_ptr<NN::Tensor> style = tensorFromImage(av[2], 512);
  std::shared_ptr<NN::Tensor> preprocessedStyle = p.forward(style);
  std::cout << "Done loading" << std::endl;

  for (int frame(1) ; frame < 240 ; ++frame)
    {
      std::cout << "Frame : " << frame << std::endl;
      std::shared_ptr<NN::Tensor> content = tensorFromImage(av[1] + std::string("/frame") + std::to_string(frame) + ".jpg", 512);
      std::shared_ptr<NN::Tensor> result = content;
      for (int i(0) ; i < 5 ; i++)
	{
	  std::shared_ptr<NN::Tensor> preprocessedContent = p.forward(result);
	  result = WCT(networks[i], preprocessedContent, preprocessedStyle);
	  result->clamp(0, 1);
	  result->mul(255);
	  saveTensorAsImage(result, "RESULTS/result_" + std::to_string(frame) + "_" + std::to_string(i) + ".png");
	  result = resizeTensor(result, 512, 288);
	  result->clamp(0, 255);
	  result = blendImage(result, content, .8f);
	}
    }

  return 0;
}
