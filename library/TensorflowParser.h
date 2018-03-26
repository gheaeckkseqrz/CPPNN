#ifndef __TENSORFLOW_PARSER_H__
#define __TENSORFLOW_PARSER_H__

#include <list>
#include "ComputeGraph.h"

namespace NN
{
  class TensorflowParser
  {
  public:
    static TensorflowParser *getInstance();
    ComputeGraph loadFromFile(std::string const &path);

    std::string getFileContent(std::string const &path);
    std::list<std::pair<std::string, std::string>> getNodeList(std::string const &s);
    std::list<std::pair<std::string, std::string>> getElementsList(std::string const &s);
    std::string getFirstElementOfType(std::string const &type, std::string const &s);
    std::list<std::pair<std::string, std::string>> getAllElementsOfType(std::string const &type, std::string s);
    std::string getAttribute(std::string const &attribute, std::string const s);
    std::list<std::pair<std::string, std::string>> getAttributeList(std::string const &s);
    std::string stripQuotes(std::string const &s);

  private:
    TensorflowParser();
    static TensorflowParser *_instance;
  };
}

#endif /* __TENSORFLOW_PARSER__ */
