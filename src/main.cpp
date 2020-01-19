

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
	VectorXd sample_vec(3),sample_vec2(3); sample_vec<<1,1,2; sample_vec2 << 0,1,1;


	std::cout<<"First row: "<< m.row(0) <<"\n";

	LossFunctions::CrossEntropy ce_loss;


	std::cout<<"CE LOSS: \n"<<ce_loss.func(sample_vec,sample_vec2)<<std::endl;


	std::cout<<"Softmax: \n"<<Normalization::softmax(sample_vec)<<std::endl;
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

	VectorXd data_vec(32); data_vec << 1,2,3,8,3,2,1,0,1,2,3,8,3,2,1,0,1,2,3,8,3,2,1,0,1,2,3,8,3,2,1,0;
	//VectorXd data_vec(8); data_vec << 1,2,3,8,3,2,1,0;

	std::shared_ptr<ANN::InputLayer> input_layer = std::make_shared<ANN::InputLayer>(32,std::make_shared<Activations::None>(),std::make_shared<Optimizers::StochasticGradientDescent>());
	std::shared_ptr<ANN::Layer> hidden_layer = std::make_shared<ANN::Layer>(16,std::make_shared<Activations::LeakyReLU>(),std::make_shared<Optimizers::StochasticGradientDescent>());
	std::shared_ptr<ANN::Layer> hidden_layer_2 = std::make_shared<ANN::Layer>(8,std::make_shared<Activations::LeakyReLU>(),std::make_shared<Optimizers::StochasticGradientDescent>());
	std::shared_ptr<ANN::Layer> hidden_layer_3 = std::make_shared<ANN::Layer>(16,std::make_shared<Activations::LeakyReLU>(),std::make_shared<Optimizers::StochasticGradientDescent>());
	std::shared_ptr<ANN::OutputLayer> output_layer = std::make_shared<ANN::OutputLayer>(32,std::make_shared<Activations::None>(),std::make_shared<Optimizers::StochasticGradientDescent>());

	input_layer->set_input_data(data_vec);

	ANN::Layer::connect_layers(input_layer,hidden_layer);
	ANN::Layer::connect_layers(hidden_layer,hidden_layer_2);
	ANN::Layer::connect_layers(hidden_layer_2,hidden_layer_3);
	ANN::Layer::connect_layers(hidden_layer_3,output_layer);

	ANN::Propagation::ForwardPropagation fp(input_layer);

	ANN::Propagation::BackwardPropagation bp(output_layer,0.001);

	VectorXd labels(32); labels << 1,2,3,8,3,2,1,0,1,2,3,8,3,2,1,0,1,2,3,8,3,2,1,0,1,2,3,8,3,2,1,0;
	//VectorXd labels(2); labels << 1,0;

	for (int i=0; i<50; i++)
	{
	input_layer->set_input_data(data_vec);
	fp.execute();
	//std::cout<<"output neurons result: \n"<<*output_layer->get_neurons_ptr()<<std::endl;
	bp.execute(labels);
	std::cout<<i<<". Error: "<<bp.get_error_val()<<std::endl;
	}


	std::cout<<"output neurons result: \n"<<*output_layer->get_neurons_ptr()<<std::endl;

	std::cout<<"weights input_layer: \n"<<*input_layer->get_output_weights_ptr()->get_weights_mat_ptr()<<std::endl;
	std::cout<<"weights hidden_layer: \n"<<*hidden_layer->get_output_weights_ptr()->get_weights_mat_ptr()<<std::endl;
	std::cout<<"weights hidden_layer_2: \n"<<*hidden_layer_2->get_output_weights_ptr()->get_weights_mat_ptr()<<std::endl;

	return 0;

}
