/*
 * Model.cpp
 *
 *  Created on: Feb 5, 2020
 *      Author: david
 */

#include "include/Model.h"

namespace SANN
{
	//constructor

    Model::Model(std::vector<uint32_t> model_by_layers_size,double learning_rate,LossFunctionPtr loss_func):
				lr_(learning_rate),
				loss_func_(loss_func),
				fp_(),
				bp_(learning_rate,loss_func)
	{
    	set_layers(model_by_layers_size);
	}

    /**
     * If model activation isn't given then default initialization is applied
     */
    void Model::set_layers(std::vector<uint32_t> model_by_layers_size,std::vector<Activations::act_t> model_activations)
    {
   	    layers_.clear();
		std::vector<ActivationFunctionPtr> act_ptr_vec;
    	if(model_activations.empty())
    	{
    		for(uint32_t i=0; i < model_by_layers_size.size(); i++)
    		{
				if(i == 0) //input size
				{
					act_ptr_vec.push_back(select_activation(act_t::ACT_NONE));
				}
				else if(i == model_by_layers_size.size()-1)//last element is the output size
				{
					act_ptr_vec.push_back(select_activation(act_t::ACT_RELU));
				}
				act_ptr_vec.push_back(select_activation(act_t::ACT_LEAKY_RELU));
    		}
    	}
    	else //activations were given
    	{
    		for(uint32_t i=0; i < model_by_layers_size.size(); i++)
			{
				if(i == 0) //input size
				{
					act_ptr_vec.push_back(select_activation(model_activations[i]));
				}
				else if(i == model_by_layers_size.size()-1)//last element is the output size
				{
					act_ptr_vec.push_back(select_activation(model_activations[i]));
				}
				act_ptr_vec.push_back(select_activation(model_activations[i]));
			}
    	}

    	for(uint32_t i=0; i < model_by_layers_size.size(); i++)
		{
			if(i == 0) //input size
			{
				std::shared_ptr<ANN::InputLayer> input_layer = std::make_shared<ANN::InputLayer>(model_by_layers_size[i],
																								 act_ptr_vec[i],
																								 std::make_shared<Optimizers::SGD>());
				layers_.push_back(input_layer);
			}
			else if(i == model_by_layers_size.size()-1)//last element is the output size
			{
				std::shared_ptr<ANN::OutputLayer> output_layer = std::make_shared<ANN::OutputLayer>(model_by_layers_size[i],
																								    act_ptr_vec[i],
																									std::make_shared<Optimizers::Adam>());

				layers_.push_back(output_layer);
			}

			layers_.push_back(std::make_shared<ANN::Layer>(model_by_layers_size[i],act_ptr_vec[i],std::make_shared<Optimizers::SGD>()));
		}
    }

    /**
	 * Assumes that layers exists
	 */
    void Model::set_activations_hidden_only(std::vector<Activations::act_t> model_activations)
    {
    	if(model_activations.size() == (layers_.size()-2))//without input and output
    	{
    		model_activations.insert(model_activations.begin(),Activations::ACT_NONE);
    		model_activations.push_back(Activations::ACT_NONE);
    		set_activations(model_activations);
    	}
    }

    /**
     * Assumes that layers exists
     */
    void Model::set_activations(std::vector<Activations::act_t> model_activations)
    {
    	if(model_activations.size() == layers_.size())
    	{
			std::list<std::shared_ptr<ANN::Layer>>::iterator it = layers_.begin();
			for ( int i=0; it != layers_.end(); it++,i++)
			{
				(*it)->set_activation_func_ptr(select_activation(model_activations[i]));
			}
    	}
    }

	/**
	 * If layer index wasn't given then set the optimizer of the last layer
	 */
	void Model::set_optimizer(Optimizers::opt_t opt_val,int layer_idx)
	{
		if(layer_idx < 0)
		{
			layers_.back()->set_optimizer(select_optimizer(opt_val));
		}
		else
		{
			std::list<std::shared_ptr<ANN::Layer>>::iterator it = std::next(layers_.begin(),layer_idx);
			(*it)->set_optimizer(select_optimizer(opt_val));
		}

	};

    bool Model::connect_layers()
    {
    	if (validate_model())
    	{
    		std::list<std::shared_ptr<ANN::Layer>>::iterator it1 = layers_.begin(),it2 = std::next(layers_.begin(),1);
    		for ( ; it2 != layers_.end(); it1++,it2++)
    		{
    			(*it1)->connect_layers(*it1,*it2);
    		}
    		layers_connected_ = true;
    		return true;
    	}
		layers_connected_ = false;
    	return false;
    }

    /**
     * Taking data row by row and labels row by row
     * data row size has to be the same as input size
     * labels row size has to be the same as output size
     */
    double Model::train(MatrixXd data,MatrixXd labels,bool print_loss)
    {
		bp_.set_params(lr_,loss_func_);
    	if(layers_connected_ == false)
    	{
    		connect_layers();
    	}
    	double last_lost = std::numeric_limits<double>::infinity();
    	for (uint32_t row = 0; row < data.rows(); row++)
    	{
    		VectorXd input_data = (data.row(row)).transpose();
    		VectorXd label_data = (labels.row(row)).transpose();
    		layers_.front()->set_new_val_to_neurons(input_data);
    		fp_.set_input_layer(layers_.front());
    		fp_.execute();
    		bp_.execute(label_data);
    		if(print_loss && (last_lost-bp_.get_error_val() > lr_))
    		{
    			std::cout<<"[cppSANN] iteration: "<<row<<" Loss val: "<<bp_.get_error_val()<<std::endl;
    		}
    		last_lost = bp_.get_error_val();
    	}
    	if(print_loss)
		{
			std::cout<<"[cppSANN] Loss val: "<<bp_.get_error_val()<<std::endl;
		}
    	return bp_.get_error_val();
    }

    MatrixXd Model::predict(MatrixXd data)
    {
    	MatrixXd prediction(data.rows(),layers_.back()->get_layer_size());
		if(layers_connected_ == false)
		{
			connect_layers();
		}
		for (uint32_t row = 0; row < data.rows(); row++)
		{
			VectorXd input_data = (data.row(row)).transpose();
			layers_.front()->set_new_val_to_neurons(input_data);
			fp_.set_input_layer(layers_.front());
			fp_.execute();
			prediction.row(row) = (*layers_.back()->get_neurons_ptr()).transpose();
		}
		return prediction;
    }


}



