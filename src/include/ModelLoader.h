/*
 * ModelLoader.h
 *
 *  Created on: Mar 22, 2020
 *      Author: david
 */

#ifndef SRC_INCLUDE_MODELLOADER_H_
#define SRC_INCLUDE_MODELLOADER_H_

#define FILE_PATH "example_model.json"

#include "Model.h"
#include <string>
#include <unordered_map>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <fstream>

using namespace boost::property_tree;
using namespace SANN;

class ModelLoader
{
	struct model_data_t_
	{
		std::vector<layer_size_t> layer_sizes;
		std::vector<act_t> activation_layers;
		std::vector<std::shared_ptr<ANN::Weights>> weights_vec;
	};

private:

	 std::shared_ptr<Model> model_ptr_;

	 void generate_weights_vec(ptree &pt,model_data_t_ &mdt);
	 void load_file(std::string &file_path,model_data_t_ &mdt);


public:
	ModelLoader();
	virtual ~ModelLoader();

	 void generate_model_from_file(std::string file_path,double lr = DEFAULT_LEARNING_RATE);

	 std::shared_ptr<Model> get_model_ptr() { return this->model_ptr_; }
};


#endif /* SRC_INCLUDE_MODELLOADER_H_ */
