import matplotlib.pyplot as plt
import numpy as np
from train import create_dataloader, train_model
from sampling_methods.random_exploration import random_exploration
from utils import combine_datasets
from metrics.one_step_pred_accuracy import compute_one_step_pred_accuracy


class ActiveLearningEvaluator():
    def __init__(self, true_env, learned_env, horizon, num_al_iterations,
                 num_epochs, batch_size, learning_rate, num_eval_repetitions=20):
        """
        Initializes the ActiveLearningEvaluator with the necessary configurations.

        Args:
            true_env: The true environment for data collection.
            learned_env: The learned environment using a dynamics model.
            horizon (T): Time horizon for trajectory sampling, referred to as T in the paper.
            num_al_iterations (N_{AL}): Number of active learning iterations, referred to as N_{AL} in the paper.
            num_epochs: Number of epochs for training the dynamics model.
            batch_size: Batch size for the training dataloader.
            learning_rate: Learning rate for the optimizer.
        """    
        self.horizon = horizon
        self.num_al_iterations = num_al_iterations
        self.true_env = true_env
        self.learned_env = learned_env
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_eval_repetitions = num_eval_repetitions

    def active_learning(self):
        """
        Implements Algorithm 1 from the paper: Actively learning dynamical systems 
        using Bayesian neural networks.
        
        - Collects initial data using random exploration.
        - Iteratively trains the dynamics model and evaluates its accuracy.
        - Updates the dataset with new trajectories after each iteration.
        - Plots the one-step predictive accuracy over iterations.
        """
        # Store accuracy results for all repetitions
        all_accuracies = []
        
        for repetition in range(self.num_eval_repetitions):
            print(f"Repetition {repetition + 1}/{self.num_eval_repetitions}")

            # Collect initial dataset using random exploration
            total_dataset = random_exploration(self.true_env, self.horizon)
            
            # Initialize a list to store one-step predictive accuracy for each iteration
            accuracy_history = []

            # Perform active learning iterations
            for iteration in range(self.num_al_iterations):
                print(f"Active Learning Iteration {iteration + 1}/{self.num_al_iterations}")
                print("Number of samples: ", len(total_dataset))
                
                # Reset the weights of the learned model
                self.learned_env.model.reset_weights()
                
                # Create a dataloader for training with the current dataset
                train_dataloader = create_dataloader(total_dataset, self.batch_size)
                
                # Train the dynamics model using the training dataloader
                train_model(self.learned_env.model, train_dataloader=train_dataloader,
                            num_epochs=self.num_epochs, learning_rate=self.learning_rate)           
                
                # Evaluate the model in inference mode
                self.learned_env.model.eval()
                
                # Compute the one-step predictive accuracy of the learned model
                one_step_pred_accuracy = compute_one_step_pred_accuracy(
                    true_env=self.true_env,
                    learned_env=self.learned_env,
                    num_samples=1250
                )
                print("One step predictive accuracy: ", one_step_pred_accuracy)
                
                # Append the accuracy to the history for plotting
                accuracy_history.append(one_step_pred_accuracy)
                
                # Collect new data using random exploration
                new_dataset = random_exploration(self.true_env, self.horizon)
                
                # Combine the new dataset with the existing total dataset
                total_dataset = combine_datasets(total_dataset, new_dataset)
        
            # Append this repetition's accuracy history to the main list
            all_accuracies.append(accuracy_history)
        
        all_accuracies = np.array(all_accuracies)
        mean_accuracies = np.mean(all_accuracies, axis=0)
        std_accuracies = np.std(all_accuracies, axis=0)

        # Plot the results
        iterations = range(1, self.num_al_iterations + 1)
        plt.figure(figsize=(10, 6))

        # Plot the mean accuracies
        plt.plot(iterations, mean_accuracies, label="Mean Accuracy", color="blue")

        # Add error bars with small caps for the standard deviation
        plt.errorbar(iterations, mean_accuracies, yerr=std_accuracies, fmt='o', 
                     ecolor='blue', capsize=3, elinewidth=1, label="±1 Standard Deviation")

        # Customize the plot
        plt.xlabel("Active Learning Iteration")
        plt.ylabel("One-Step Predictive Accuracy")
        plt.title("Mean and Standard Deviation of Accuracy Across Active Learning Iterations")
        plt.legend()
        plt.grid(True)
        plt.show()