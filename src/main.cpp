

#include "Model.h"
#include "ModelLoader.h"


int main(int ac, char** av);
/**
 * Examples how to use
 */



int main(int ac, char** av)
{
	MatrixXd data_mat(10,8); data_mat << 1,2,3,8,3,2,1,0,
										1,2,3,8,3,2,1,0,
										1,2,3,8,3,2,1,0,
										1,2,3,8,3,2,1,0,
										1,2,3,8,3,2,1,0,
										1,2,3,8,3,2,1,0,
										1,2,3,8,3,2,1,0,
										1,2,3,8,3,2,1,0,
										1,2,3,8,3,2,1,0,
										1,2,3,8,3,2,1,0;
	MatrixXd label_mat(10,2); label_mat <<0,1,
										  0,1,
										  0,1,
										  0,1,
										  0,1,
										  0,1,
										  0,1,
										  0,1,
										  0,1,
										  0,1;
	MatrixXd data_with_noise = data_mat;
	data_with_noise += ExtMath::randn(10,8);

	std::vector<uint32_t> layers_sizes{8,4,3,2};
	std::vector<act_t> act_types_vec{act_t::ACT_NONE,act_t::ACT_ELU,act_t::ACT_ELU,act_t::ACT_LEAKY_RELU,act_t::ACT_LEAKY_RELU,act_t::ACT_NONE};
	SANN::Model model(layers_sizes,0.1);
	model.set_activations(act_types_vec);
	model.train(data_mat,label_mat,true);

	MatrixXd results = model.predict(data_with_noise);
	std::cout<<"data with noise: \n"<<data_with_noise<<std::endl;
	std::cout<<"results: \n"<<results<<std::endl;

	ModelLoader ml;
	ml.generate_model_from_file(FILE_PATH);
	std::shared_ptr<Model> loaded_model = ml.get_model_ptr();

	std::cout<<"Loaded model: "<<loaded_model->get_list_of_layers().front()->get_layer_size()<<std::endl;

	std::cout<<"Done"<<std::endl;

	return 0;

}
