#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include "ComputeGraph.h"
#include "TensorflowFactory.h"
#include "TensorflowParser.h"

namespace NN
{
  TensorflowParser *TensorflowParser::_instance = nullptr;

  TensorflowParser *TensorflowParser::getInstance()
  {
    if (_instance == nullptr)
      _instance = new TensorflowParser();
    return _instance;
  }

  TensorflowParser::TensorflowParser()
  {
  }

  ComputeGraph TensorflowParser::loadFromFile(std::string const &path)
  {
    std::string fileContent = getFileContent(path);
    std::list<std::pair<std::string, std::string>> nodeList = getNodeList(fileContent);
    NN::ComputeGraph graph;
    for (std::pair<std::string, std::string> e : nodeList)
      graph.add(e.first, TensorflowFactory::getInstance()->createNode(e.second));
    return graph;
  }

  std::string TensorflowParser::getFileContent(std::string const &path)
  {
    std::FILE *fp = std::fopen(path.c_str(), "rb");
    if (fp)
      {
	std::string contents;
	std::fseek(fp, 0, SEEK_END);
	contents.resize(std::ftell(fp));
	std::rewind(fp);
	std::fread(&contents[0], 1, contents.size(), fp);
	std::fclose(fp);
	return contents;
      }
    return "";
  }

  std::list<std::pair<std::string, std::string>> TensorflowParser::getNodeList(std::string const &s)
  {
    std::list<std::pair<std::string, std::string>> nodeList;
    std::string graph_def = getFirstElementOfType("graph_def", s);
    std::list<std::pair<std::string, std::string>> l = getElementsList(graph_def);
    for (std::pair<std::string, std::string> t : l)
      {
	if (t.first == "node")
	  nodeList.push_back(std::pair<std::string, std::string>(stripQuotes(getAttribute("name", t.second)), t.second));
      }
    return nodeList;
  }

  std::list<std::pair<std::string, std::string>> TensorflowParser::getElementsList(std::string const &s)
  {
    std::list<std::pair<std::string, std::string>> l;

    int keyStartIndex = -1;
    int valStartIndex = -1;
    int depth = 0;
    bool quoted = false;

    std::string key;
    std::string val;

    for (int i = 0 ; i < s.size() ; ++i)
      {
	if (s[i] == ' ' || s[i] == '\n' || s[i] == '\t')
	  {
	    if (s[i] == '\n' && depth == 0)
	      keyStartIndex = -1;
	  }
	else if (s[i] == '"' && s[i-1] != '\\')
	  {
	    quoted = !quoted;
	  }
	else if (s[i] == '{')
	  {
	    if (depth == 0)
	      {
		key = s.substr(keyStartIndex, i - keyStartIndex - 1);
		valStartIndex = i;
	      }
	    if (!quoted)
	      depth++;
	  }
	else if (s[i] == '}')
	  {
	    if (!quoted)
	      depth--;
	    if (depth == 0 && !quoted)
	      {
		val = s.substr(valStartIndex + 1, i - valStartIndex - 1);
		keyStartIndex = -1;
		valStartIndex = -1;
		l.push_back(std::pair<std::string, std::string>(key, val));
	      }
	  }
	else if (keyStartIndex == -1)
	  {
	    keyStartIndex = i;
	  }
      }
    return l;
  }

  std::string TensorflowParser::getFirstElementOfType(std::string const &type, std::string const &s)
  {
    std::list<std::pair<std::string, std::string>> l = getElementsList(s);
    for (std::pair<std::string, std::string> i : l)
      {
	if (i.first == type)
	  return i.second;
      }
    std::cerr << "Can't find element " << type << std::endl;
    return "";
  }

  std::list<std::pair<std::string, std::string>> TensorflowParser::getAllElementsOfType(std::string const &type, std::string s)
  {
    std::list<std::pair<std::string, std::string>> l = getElementsList(s);
    std::remove_if(l.begin(), l.end(), [&type](std::pair<std::string, std::string> x){ return x.first != type; });
    return l;
  }

  std::string TensorflowParser::getAttribute(std::string const &attribute, std::string const s)
  {
    std::list<std::pair<std::string, std::string>> l = getAttributeList(s);
    for (std::pair<std::string, std::string> t : l)
      if (t.first == attribute) return stripQuotes(t.second);

    std::cerr << "Can't find attribute [" << attribute << "]" << std::endl;
    return "";
  }

  std::list<std::pair<std::string, std::string>> TensorflowParser::getAttributeList(std::string const &e)
  {
    bool quoted = false;
    int depth = 0;
    int keyStartIndex = -1;
    int valStartIndex = -1;
    std::string key = "";
    std::string val = "";
    std::list<std::pair<std::string, std::string>> l;

    for (int i = 0 ; i < e.size() ; ++i)
      {
	if (e[i] == '\"' && i > 0 && e[i-1] != '\\')
	  {
	    quoted = !quoted;
	  }
	if (e[i] == '{' && !quoted)
	  {
	    depth++;
	    keyStartIndex = -1;
	  }
	else if (e[i] == '}' && ! quoted)
	  {
	    depth--;
	  }
	else if (e[i] == ' ' || e[i] == '\n' || e[i] == '\t' || e[i] == '\r')
	  {
	    if (valStartIndex >= 0 && !quoted)
	      {
		val = stripQuotes(e.substr(valStartIndex, i - valStartIndex));
		l.push_back(std::pair<std::string, std::string>(stripQuotes(key), val));
		key = "";
		val = "";
		keyStartIndex = -1;
		valStartIndex = -1;
	      }
	  }
	else if (depth == 0 && keyStartIndex == -1 && !quoted)
	  {
	    keyStartIndex = i;
	  }
	else if (keyStartIndex >= 0 && e[i] == ':' && !quoted)
	  {
	    key = e.substr(keyStartIndex, i - keyStartIndex);
	  }
	else if (valStartIndex == -1 && key != "")
	  {
	    valStartIndex = i;
	  }
      }
    return l;
  }

  std::string TensorflowParser::stripQuotes(std::string const &s)
  {
    if (s[0] == '"' && s[s.size() - 1] == '"')
      return s.substr(1, s.size() - 2);
    return s;
  }
}
