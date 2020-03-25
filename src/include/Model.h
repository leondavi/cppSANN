/*
 * Model.h
 *
 *  Created on: Feb 5, 2020
 *      Author: david leon
 */

#ifndef SRC_INCLUDE_MODEL_H_
#define SRC_INCLUDE_MODEL_H_


#include "propagation.h"
#include <list>
#include <string>
#include <vector>


namespace SANN
{
	/**
	 * An abstract model class
	 */
	class Model
	{
	private:

	    double lr_;
	    std::list<std::shared_ptr<ANN::Layer>> layers_;
	    LossFunctionPtr loss_func_;

	    ANN::Propagation::ForwardPropagation fp_;
	    ANN::Propagation::BackwardPropagation bp_;

	    bool layers_connected_;

	    bool connect_layers();

	    void generate_act_vec(std::vector<Activations::act_t> hidden_activations,
							 std::vector<ActivationFunctionPtr> &act_ptr_vec_out,
							 act_t input = ACT_NONE, act_t output = ACT_NONE);

	    bool generate_layers_from_weights(std::vector<std::shared_ptr<ANN::Weights>> weights_vec,
	       							      std::vector<ActivationFunctionPtr> &act_ptr_vec,
	   								      Optimizers::opt_t optimizer = Optimizers::OPT_ADAM);

	public:

	    Model(std::vector<uint32_t> model_by_layers_size,
	    		double learning_rate = DEFAULT_LEARNING_RATE,
				LossFunctionPtr loss_func = std::make_shared<LossFunctions::MSELoss>());

	    Model(std::vector<layer_size_t> model_by_layers_size,
				std::vector<act_t> activation_layers,
				std::vector<std::shared_ptr<ANN::Weights>> weights_vec,
				double learning_rate = DEFAULT_LEARNING_RATE,
				Optimizers::opt_t optimizer = Optimizers::OPT_ADAM,
				LossFunctionPtr loss_func = std::make_shared<LossFunctions::MSELoss>());

		Model(double learning_rate = DEFAULT_LEARNING_RATE,LossFunctionPtr loss_func = std::make_shared<LossFunctions::MSELoss>()) :
			lr_(learning_rate),loss_func_(loss_func),fp_(),bp_(learning_rate,loss_func),layers_connected_(false)
	    {}
		Model(std::list<std::shared_ptr<ANN::Layer>> layers,
			  double learning_rate = DEFAULT_LEARNING_RATE,
			  LossFunctionPtr loss_func = std::make_shared<LossFunctions::MSELoss>()) :
			  lr_(learning_rate),
			  layers_(layers),
			  loss_func_(loss_func),
			  fp_(),
			  bp_(learning_rate,loss_func),
			  layers_connected_(false)
		{}
		virtual ~Model() {};

		virtual bool validate_model() { return layers_.size() >= 2; } //at least two layers to create a valid model

		std::list<std::shared_ptr<ANN::Layer>> get_list_of_layers() { return this->layers_; }

		virtual double train(MatrixXd data,MatrixXd labels,bool print_loss = false); //return value of final loss
	    MatrixXd predict(MatrixXd data); //returns the predictions - each row is a single prediction


		//setters
	    void set_layers(std::vector<uint32_t> model_by_layers_size,std::vector<Activations::act_t> model_activations = std::vector<Activations::act_t>());
	    void set_activations(std::vector<Activations::act_t> model_activations = std::vector<Activations::act_t>());
	    void set_activations_hidden_only(std::vector<Activations::act_t> model_activations);

		void set_learning_rate(double learning_rate) {this->lr_ = learning_rate; }
		void set_optimizer(Optimizers::opt_t opt_val,int layer_idx = -1);


		std::vector<std::shared_ptr<ANN::Weights>> get_weights_of_model();
		std::vector<VectorXd> get_neurons_of_model();

		void load_weights_of_model(std::string path);//TODO

		void save_model_to_file();//TODO
		void load_model_from_file();//TODO

	};

	class Autoencoder : public Model
	{
	private:

	public:
		Autoencoder(double learning_rate = DEFAULT_LEARNING_RATE) : Model(learning_rate) {};
		Autoencoder(std::vector<uint32_t> model_by_layers_size,
				    double learning_rate = DEFAULT_LEARNING_RATE,
				    std::vector<Activations::act_t> model_activations = std::vector<Activations::act_t>()) : Model(learning_rate)
		            {
						set_layers(model_by_layers_size,model_activations);
		            };


		bool validate_model() override { return (get_list_of_layers().size() >= 3) &&
												(get_list_of_layers().front()->get_layer_size() == get_list_of_layers().back()->get_layer_size());}

	};

	class CustomModel : public Model
	{
		private:

		public:
			CustomModel(std::list<std::shared_ptr<ANN::Layer>> list_of_layers, double learning_rate = DEFAULT_LEARNING_RATE) :
				        Model(list_of_layers,learning_rate) {}

	};

}

#endif /* SRC_INCLUDE_MODEL_H_ */
