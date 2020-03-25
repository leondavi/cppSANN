/*
 * ModelLoader.cpp
 *
 *  Created on: Mar 22, 2020
 *      Author: david
 */

#include <ModelLoader.h>



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

		for (auto& elem : pt.get_child("activations"))
		{
			std::cout<<elem.second.get_value<std::string>()<<",";
			mt.activation_layers.push_back(Activations::str_to_act_t(elem.second.get_value<std::string>()));
		}

		std::cout<<"dims list ";
		for (auto& elem : pt.get_child("layers"))
		{
			std::cout<<elem.second.get_value<unsigned int>()<<","<<std::endl;
			mt.layer_sizes.push_back(elem.second.get_value<unsigned int>());
		}

		// ---- generate matrices -----//
		std::vector<std::shared_ptr<ANN::Weights>> weights_vec;
		generate_weights_vec(pt,mt,weights_vec);
		std::cout<<"loaded"<<std::endl;
		//std::string layers = pt.get_child("layers").get_value<std::string>();
		//std::string activations = pt.get<std::string>("activations");
		//std::string a = "";

	} catch(...)
	{
		std::cout<<"[cppSANN] ModelLoader has got an invalid json file!"<<std::endl;
	}
}

void ModelLoader::generate_weights_vec(ptree &pt,meta_data_t_ &mt,std::vector<std::shared_ptr<ANN::Weights>> &vec_of_weights)
{
	std::vector <Eigen::MatrixXd> weights_vec; weights_vec.resize(mt.layer_sizes.size()-1);
	std::vector <Eigen::VectorXd> bias_vec; bias_vec.resize(mt.layer_sizes.size()-1);

	int w = 0;
	int b = 0;
	enum {WEIGHTS,BIAS};
	for (auto &layer : pt.get_child("weights"))
	{
		std::cout<<"Layer: "<<layer.first<<std::endl;
		int case_select = (layer.first.find("weight") != std::string::npos) ? WEIGHTS : BIAS;
		switch (case_select)
		{
		  case WEIGHTS: {weights_vec[w] = Eigen::MatrixXd(mt.layer_sizes[w+1],mt.layer_sizes[w]); break;}
		  case BIAS:    {bias_vec[b] = Eigen::VectorXd(mt.layer_sizes[b+1]); break;}
		}

		int i = 0;
		for (auto &row : layer.second)
		{
			switch (case_select)
			{
				case WEIGHTS:	{
					int j = 0;
					for (auto &cell : row.second)
					{
						weights_vec[w](i,j) = cell.second.get_value<double>();
						j++;
					}
					break;
				}
				case BIAS:     {
					bias_vec[b](i) = row.second.get_value<double>(); // @suppress("Field cannot be resolved")
					break;
				}
			}//end of switch case
			i++;
		}
		switch (case_select)
		{
		case WEIGHTS: {w++; break;}
		case BIAS: {b++; break;}
		}
	}

	for (unsigned i=0; i < weights_vec.size(); i++)
	{
		std::shared_ptr<ANN::Weights> weights_ptr = std::make_shared<ANN::Weights>(weights_vec[i],bias_vec[i]);
	}
}
