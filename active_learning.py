import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from train import create_dataloader, train_model
from sampling_methods.sampling_method import SamplingMethod
from sampling_methods.random_exploration import RandomExploration
from utils import combine_datasets
from metrics.one_step_pred_accuracy import compute_one_step_pred_accuracy
from typing import List

class ActiveLearningEvaluator():
    def __init__(
            self, true_env: gym.Env, learned_env: gym.Env, sampling_methods: List[SamplingMethod],
            num_al_iterations: int, num_epochs: int, batch_size: int,
            learning_rate: float, num_eval_repetitions: int = 20
        ) -> None:
        """
        Initializes the ActiveLearningEvaluator with the necessary configurations.

        Args:
            true_env (gym.Env): The true environment for data collection.
            learned_env (gym.Env): The learned environment using a dynamics model.
            sampling_methods (List[SamplingMethod]): List of methods to collect (state, action, next state)
                                                     pairs.
            num_al_iterations (N_{AL}) (int): Number of active learning iterations.
            num_epochs (int): Number of epochs for training the dynamics model.
            batch_size (int): Batch size for the training dataloader.
            learning_rate (float): Learning rate for the optimizer.
            num_eval_repetitions (int): Number of evaluation repetitions to compute robust metrics.
        """    
        self.true_env = true_env
        self.learned_env = learned_env
        self.sampling_methods = sampling_methods
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
        # Store mean and standard deviation of the accuracies for all sampling methods
        all_mean_accuracies = []
        all_std_accuracies = []
        for sampling_method in self.sampling_methods:
            print(f"Starting active learning with the sampling method: '{sampling_method.name}'.")

            # Store accuracy results of the current sampling method over all repetitions
            sampling_method_accuracies = []
            
            for repetition in range(self.num_eval_repetitions):
                print(f"Evaluation Repetition {repetition + 1}/{self.num_eval_repetitions}")

                # Collect initial dataset using random exploration
                random_exploration = RandomExploration(sampling_method.horizon)
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
                        new_dataset = sampling_method.sample(self.true_env, self.learned_env)
                        
                        # Merge the new dataset with the existing dataset
                        total_dataset = combine_datasets(total_dataset, new_dataset)
            
                # Append this repetition's accuracy history to the main list
                sampling_method_accuracies.append(accuracy_history)
        
            sampling_method_accuracies = np.array(sampling_method_accuracies)
            all_mean_accuracies.append(np.mean(sampling_method_accuracies, axis=0))
            all_std_accuracies.append(np.std(sampling_method_accuracies, axis=0))

        # Plot the results
        self.plot_predictive_accuracies(all_mean_accuracies=all_mean_accuracies,
                                        all_std_accuracies=all_std_accuracies)

    def plot_predictive_accuracies(
            self, all_mean_accuracies: List[List[float]], all_std_accuracies: List[List[float]]
    ) -> None:
        """
        Plots the mean and standard deviation of predictive accuracies for different sampling
        methods over active learning iterations.

        Args:
            all_mean_accuracies (List[List[float]]): A list where each element is a list of mean 
                accuracies for a specific sampling method across active learning iterations.
                Shape: [num_methods, num_al_iterations].
            all_std_accuracies (List[List[float]]): A list where each element is a list of standard 
                deviations for a specific sampling method across active learning iterations.
                Shape: [num_methods, num_al_iterations].
        """
        
        # Define map of colors to distinguish the sampling methods
        color_map = {
            "Random Exploration": "blue",
            "Random Sampling Shooting": "green",
        }
        
        iterations = range(1, self.num_al_iterations + 1)

        plt.figure(num="One Step Predictive Accuracy over Active Learning Iterations", figsize=(10, 6))
                
        # Plot mean and std for each sampling method using the specified color
        for i, sampling_method in enumerate(self.sampling_methods):
            method_name = sampling_method.name
            mean_accuracies = all_mean_accuracies[i]
            std_accuracies = all_std_accuracies[i]

            # Lookup the color for this method from the map
            color = color_map.get(method_name, "black")  # default to black if not found

            plt.plot(iterations, mean_accuracies, label=f"{method_name}", color=color)
            plt.errorbar(iterations, mean_accuracies, yerr=std_accuracies, fmt='o', 
                         color=color, ecolor=color, capsize=3, elinewidth=1)

        # Customize and show the plot
        plt.xlabel("Active Learning Iteration")
        plt.ylabel("One-Step Predictive Accuracy")
        plt.title("One-Step Predictive Accuracy over Active Learning Iterations")
        plt.legend()
        plt.grid(True)
        plt.show()