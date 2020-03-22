/*
 * ModelLoader.h
 *
 *  Created on: Mar 22, 2020
 *      Author: david
 */

#ifndef SRC_INCLUDE_MODELLOADER_H_
#define SRC_INCLUDE_MODELLOADER_H_

#define FILE_PATH "/home/david/workspace/cppSANN/example_model.sann"

#include <Model.h>
#include <string>

using namespace SANN;

class ModelLoader {
	std::string model_file_path_;

public:
	ModelLoader(std::string model_file_path) : model_file_path_(model_file_path) {}
	virtual ~ModelLoader();

private:


};

#endif /* SRC_INCLUDE_MODELLOADER_H_ */
