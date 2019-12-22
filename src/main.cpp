

#include "include/definitions.hh"
#include "include/Layer.h"
#include "include/activation_functions.h"
#include "include/propagation.h"
#include "include/loss_functions.h"

int main(int ac, char** av);
/**
 * Examples how to use
 */

int main(int ac, char** av)
{
	MatrixXd m(4,3);
	m << 1, 2, 3,
	     4, 5, 6,
	     7, 8, 9,
		 10,11,12;
	ANN::Weights new_weight(m);
	LossFunctions::CategoricalCrossEntropyLoss lgloss;
	VectorXd sample_vec(4),sample_vec2(4); sample_vec<<1,1,0,0; sample_vec2 << 0,1,0,1;

	std::cout<<"Softmax: \n"<<Normalization::softmax(sample_vec)<<std::endl;
	std::cout<<"logloss: \n"<<lgloss.func(sample_vec,sample_vec2)<<std::endl;
	std::cout<<"Mse results: "<<ExtMath::mse(sample_vec,sample_vec2)<<std::endl;

	//std::cout<<"Before: "<<*new_weight.get_weights_mat()<<std::endl;
	VectorXd dprod_vec = new_weight.dot(sample_vec)*100;
	std::cout<<"before dprod vec: \n"<<dprod_vec<<std::endl;

	//std::cout<<"dprod vec: \n"<<dprod_vec<<std::endl;

	Activations::Sigmoid sigmoid;
	dprod_vec = dprod_vec.unaryExpr(sigmoid.get_func());


//	std::cout<<"After: "<<*new_weight.get_weights_mat()<<std::endl;
//	std::cout<<"sample_vec: \n"<<sample_vec<<std::endl;
	std::cout<<"dprod vec: \n"<<dprod_vec<<std::endl;


	std::cout<<"forward propagation testing: "<<std::endl;

	VectorXd data_vec(8); data_vec << 1,2,3,4,3,2,1,0;
	std::shared_ptr<ANN::InputLayer> input_layer = std::make_shared<ANN::InputLayer>(8);
	std::shared_ptr<ANN::Layer> hidden_layer = std::make_shared<ANN::Layer>(6);
	std::shared_ptr<ANN::OutputLayer> output_layer = std::make_shared<ANN::OutputLayer>(4);

	input_layer->set_input_data(data_vec);

	ANN::Layer::connect_layers(input_layer,hidden_layer);
	ANN::Layer::connect_layers(hidden_layer,output_layer);

	ANN::Propagation::ForwardPropagation fp(input_layer);

	fp.execute();
	std::cout<<"output neurons result: \n"<<*output_layer->get_neurons_ptr()<<std::endl;

	return 0;

}
