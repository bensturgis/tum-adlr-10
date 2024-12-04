import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from train import create_dataloader, train_model
from sampling_methods.sampling_method import SamplingMethod
from sampling_methods.random_exploration import RandomExploration
from utils import combine_datasets
from metrics.one_step_pred_accuracy import compute_one_step_pred_accuracy
import typing

class ActiveLearningEvaluator():
    def __init__(
            self, true_env: gym.Env, learned_env: gym.Env, sampling_method: SamplingMethod,
            num_al_iterations: int, num_epochs: int, batch_size: int,
            learning_rate: float, num_eval_repetitions: int = 20
        ) -> None:
        """
        Initializes the ActiveLearningEvaluator with the necessary configurations.

        Args:
            true_env (gym.Env): The true environment for data collection.
            learned_env (gym.Env): The learned environment using a dynamics model.
            sampling_method (SamplingMethod): Method to collect (state, action, next state) pairs.
            num_al_iterations (N_{AL}) (int): Number of active learning iterations.
            num_epochs (int): Number of epochs for training the dynamics model.
            batch_size (int): Batch size for the training dataloader.
            learning_rate (float): Learning rate for the optimizer.
            num_eval_repetitions (int): Number of evaluation repetitions to compute robust metrics.
        """    
        self.true_env = true_env
        self.learned_env = learned_env
        self.sampling_method = sampling_method
        self.num_al_iterations = num_al_iterations
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_eval_repetitions = num_eval_repetitions

    def active_learning(self) -> None:
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
            print(f"Evaluation Repetition {repetition + 1}/{self.num_eval_repetitions}")

            # Collect initial dataset using random exploration
            random_exploration = RandomExploration(self.sampling_method.horizon)
            total_dataset = random_exploration.sample(self.true_env)
            
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
                train_model(model=self.learned_env.model, train_dataloader=train_dataloader,
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
                
                # Collect new data only if further training iterations remain
                if iteration < self.num_al_iterations - 1:
                    # Sample new data using the specified sampling method
                    new_dataset = self.sampling_method.sample(self.true_env, self.learned_env)
                    
                    # Merge the new dataset with the existing dataset
                    total_dataset = combine_datasets(total_dataset, new_dataset)
        
            # Append this repetition's accuracy history to the main list
            all_accuracies.append(accuracy_history)
        
        all_accuracies = np.array(all_accuracies)
        mean_accuracies = np.mean(all_accuracies, axis=0)
        std_accuracies = np.std(all_accuracies, axis=0)

        # Plot the results
        iterations = range(1, self.num_al_iterations + 1)
        plt.figure(num="One Step Predictive Accuracy over Active Learning Iterations", figsize=(10, 6))

        # Plot the mean accuracies
        plt.plot(iterations, mean_accuracies, label="Mean Accuracy", color="blue")

        # Add error bars with caps for the standard deviation
        plt.errorbar(iterations, mean_accuracies, yerr=std_accuracies, fmt='o', 
                     ecolor='blue', capsize=3, elinewidth=1, label="±1 Standard Deviation")

        # Customize the plot
        plt.xlabel("Active Learning Iteration")
        plt.ylabel("One-Step Predictive Accuracy")
        plt.title("One Step Predictive Accuracy over Active Learning Iterations")
        plt.legend()
        plt.grid(True)
        plt.show()