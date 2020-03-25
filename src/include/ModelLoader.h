/*
 * ModelLoader.h
 *
 *  Created on: Mar 22, 2020
 *      Author: david
 */

#ifndef SRC_INCLUDE_MODELLOADER_H_
#define SRC_INCLUDE_MODELLOADER_H_

#define FILE_PATH "/home/david/workspace/cppSANN/example_model.json"

#include <Model.h>
#include <string>
#include <unordered_map>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <fstream>

using namespace boost::property_tree;
using namespace SANN;

class ModelLoader
{
	struct meta_data_t_
	{
		std::vector<int> layer_sizes;
		std::vector<act_t> activation_layers;
	};

private:
	 std::unordered_map<std::string,Model> file_to_model_map_;

	 void generate_weights_vec(ptree &pt,meta_data_t_ &mt,std::vector<std::shared_ptr<ANN::Weights>> &vec_of_weights);

public:
	ModelLoader();
	virtual ~ModelLoader();

	 void load_file(std::string file_path);



};


#endif /* SRC_INCLUDE_MODELLOADER_H_ */
