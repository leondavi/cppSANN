/*
 * ModelLoader.cpp
 *
 *  Created on: Mar 22, 2020
 *      Author: david
 */

#include <ModelLoader.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <fstream>

using namespace boost::property_tree;

ModelLoader::ModelLoader() {
	// TODO Auto-generated constructor stub



}

ModelLoader::~ModelLoader() {
	// TODO Auto-generated destructor stub
}

void ModelLoader::load_file(std::string file_path)
{
	try
	{
		std::ifstream json_ifs(file_path);
		ptree pt;
		json_parser::read_json(json_ifs, pt); //load json file
		meta_data_t_ mt;
		for (auto & node : pt)
		{
			if (node.first == "activations")
			{
				std::cout<<"activations ";
				for (auto& elem : node.second)
				{
					std::cout<<elem.second.get_value<std::string>()<<",";
				}
				std::cout<<std::endl;
			}
			else if (node.first == "dims list")
			{
				std::cout<<"dims list ";
				for (auto& elem : node.second)
				{
					std::cout<<elem.second.get_value<std::string>()<<",";
				}
				std::cout<<std::endl;
			}
			else //collect data
			{
				std::cout<<"data "<<node.first;
				std::cout<<std::endl;
			}
		}
		//std::string layers = pt.get_child("layers").get_value<std::string>();
		//std::string activations = pt.get<std::string>("activations");
		//std::string a = "";



	} catch(...)
	{
		std::cout<<"[cppSANN] ModelLoader has got an invalid json file!"<<std::endl;
	}
}
