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

public:
	ModelLoader();
	virtual ~ModelLoader();

	 void load_file(std::string file_path);



};

#endif /* SRC_INCLUDE_MODELLOADER_H_ */
