import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from train import train_model
from sampling_methods.sampling_method import SamplingMethod
from sampling_methods.random_exploration import RandomExploration
from utils.train_utils import combine_datasets, create_dataloader, create_test_dataset
from metrics.one_step_pred_accuracy import OneStepPredictiveAccuracyEvaluator
from typing import Dict, List
import pandas as pd
from pathlib import Path
import json
import torch

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


    def active_learning(self, show: bool = True, save: bool = True) -> None:
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

        # Dictionary to store collected state trajectories across methods and runs
        # state_trajectories[sampling_method_name][repetition] -> np.ndarray of shape:
        # (num_al_iterations, horizon, state_dim)
        state_trajectories = {}

        # Dictionary to store training and iteration-level results
        # training_results[sampling_method_name][repetition][iteration] = {
        #     "train_loss": [... per epoch ...],
        #     "test_loss": [... per epoch ...],
        #     "model_weights": state_dict()
        # }
        training_results = {}

        # Get the state dimensionality
        state_dim = self.true_env.observation_space.shape[0]

        # Extract the horizon from the sampling methods
        horizon = self.sampling_methods[0].horizon
        for sampling_method in self.sampling_methods:
            assert horizon == sampling_method.horizon, (
                "All sampling methods need to have the same horizon for comparison."
            )
        
        # Extract the minimum and maximum state values the environment can reach for the given horizon
        state_bounds = self.true_env.compute_state_bounds(horizon=horizon)

        # Create dataset and dataloader of (state, action, next_state) samples from the true environment
        test_dataset = create_test_dataset(true_env=self.true_env, num_samples=1250, state_bounds=state_bounds)
        test_dataloader = create_dataloader(dataset=test_dataset, batch_size=self.batch_size)

        # Initialize one-step predictive accuracy evaluator with the learned environment and the dataset
        one_step_pred_acc_eval = OneStepPredictiveAccuracyEvaluator(learned_env=self.learned_env,
                                                                    dataset=test_dataset)

        for sampling_method in self.sampling_methods:
            print("--------------------------------------------------------------------------------------")
            print(f"Starting active learning with the sampling method: '{sampling_method.name}'.")
            print("--------------------------------------------------------------------------------------")

            # Store accuracy results of the current sampling method over all repetitions
            sampling_method_accuracies = []

            # Initialize structures for state trajectories and training results for this method
            state_trajectories[sampling_method.name] = {}
            training_results[sampling_method.name] = {}
            
            for repetition in range(self.num_eval_repetitions):
                print(f"Evaluation Repetition {repetition + 1}/{self.num_eval_repetitions}")

                # Collect initial dataset using random exploration
                random_exploration = RandomExploration(sampling_method.horizon)
                total_dataset = random_exploration.sample(self.true_env)
                
                # Initialize a list to store one-step predictive accuracy for each iteration
                accuracy_history = []

                # Initialize training results for this repetition
                training_results[sampling_method.name][repetition] = {}

                # Initialize the numpy array for saving state trajectories for all iterations in the repetition
                # Shape: (num_al_iterations, horizon, state_dim)
                state_trajectories[sampling_method.name][repetition] = np.zeros(
                    (self.num_al_iterations, sampling_method.horizon, state_dim)
                )
                state_trajectory = total_dataset.tensors[0].numpy()  # shape: (horizon, state_dim)
                state_trajectories[sampling_method.name][repetition][0, :, :] = state_trajectory

                # Perform active learning iterations
                for iteration in range(self.num_al_iterations):
                    print(f"Active Learning Iteration {iteration + 1}/{self.num_al_iterations}")
                    print("Number of samples: ", len(total_dataset))
                    
                    # Reset the weights of the learned model
                    self.learned_env.model.reset_weights()
                    
                    # Create a dataloader for training with the current dataset
                    train_dataloader = create_dataloader(total_dataset, self.batch_size)
                    
                    # Train the dynamics model using the training dataloader
                    train_loss, test_loss, model_weights = train_model(model=self.learned_env.model,
                                                                       train_dataloader=train_dataloader,
                                                                       test_dataloader=test_dataloader,
                                                                       num_epochs=self.num_epochs,
                                                                       learning_rate=self.learning_rate)           
                                    
                    # Store the iteration-level results (losses and model weights)
                    training_results[sampling_method.name][repetition][iteration] = {
                        "train_loss": train_loss,
                        "test_loss": test_loss,
                        "model_weights": model_weights
                    }
                    
                    # Evaluate the model in inference mode
                    self.learned_env.model.eval()
                    
                    # Compute the one-step predictive accuracy of the learned model
                    one_step_pred_accuracy = one_step_pred_acc_eval.compute_one_step_pred_accuracy()
                    
                    print("One step predictive accuracy: ", one_step_pred_accuracy)
                    
                    # Append the accuracy to the history for plotting
                    accuracy_history.append(one_step_pred_accuracy)
                    
                    # Collect new data only if further training iterations remain
                    if iteration < self.num_al_iterations - 1:
                        # Sample new data using the specified sampling method
                        new_dataset = sampling_method.sample(self.true_env, self.learned_env)
                        
                        # Merge the new dataset with the existing dataset
                        total_dataset = combine_datasets(total_dataset, new_dataset)
            
                        # Store the newly sampled state trajectory
                        state_trajectory = new_dataset.tensors[0].numpy()  # shape: (horizon, state_dim)
                        state_trajectories[sampling_method.name][repetition][iteration + 1, :, :] = state_trajectory


                # Append this repetition's accuracy history to the main list
                sampling_method_accuracies.append(accuracy_history)
        
            sampling_method_accuracies = np.array(sampling_method_accuracies)
            all_mean_accuracies.append(np.mean(sampling_method_accuracies, axis=0))
            all_std_accuracies.append(np.std(sampling_method_accuracies, axis=0))

        # Plot and save the results
        if show or save:
            self.create_active_learning_plot(all_mean_accuracies=all_mean_accuracies,
                                             all_std_accuracies=all_std_accuracies)
        
        if save:
            self.save_active_learning_results(all_mean_accuracies=all_mean_accuracies,
                                              all_std_accuracies=all_std_accuracies,
                                              state_trajectories=state_trajectories,
                                              training_results=training_results)
        
        if show:
            plt.show()


    def params_to_dict(self) -> Dict[str, str]:
        """
        Converts hyperparameters into a dictionary.
        """
        parameter_dict = {
            "true_env": self.true_env.params_to_dict(),
            "learned_env": self.learned_env.params_to_dict(),
            "sampling_methods": [
                sampling_method.params_to_dict() for sampling_method in self.sampling_methods
            ],
            "num_eval_repetitions": self.num_eval_repetitions,
            "num_al_iterations": self.num_al_iterations,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "metric": "One-step predicitive accuracy"
        }
        return parameter_dict


    def save_active_learning_results(
            self, all_mean_accuracies: List[List[float]],
            all_std_accuracies: List[List[float]],
            state_trajectories: Dict[str, Dict[int, np.ndarray]],
            training_results: Dict[str, Dict[int, List[Dict]]]
    ) -> None:
        """
        Saves the results of the active learning experiment, including:
        - A summary plot of one-step predictive accuracies over iterations.
        - Mean and standard deviation of predictive accuracies for each sampling method.
        - Hyperparameters of the experiment.
        - State trajectories collected during the experiment.
        - Iteration-level training results (train/test losses, model weights, and a loss plot).

        Args:
            all_mean_accuracies (List[List[float]]): A list where each element is a list of mean 
                accuracies for a specific sampling method across active learning iterations.
                Shape: [num_methods, num_al_iterations].
            all_std_accuracies (List[List[float]]): A list where each element is a list of standard 
                deviations for a specific sampling method across active learning iterations.
                Shape: [num_methods, num_al_iterations].
            state_trajectories (Dict[str, Dict[int, Dict[int, np.ndarray]]]): A nested dictionary storing 
                state trajectories, where the keys are sampling method names, repetitions, 
                and iterations.
            training_results (Dict[str, Dict[int, List[Dict]]]):
                A nested dictionary containing iteration-level results for each sampling method and repetition.
        """
        # Define the base directory for experiment results
        base_dir = Path(__file__).parent / "experiments" / "active_learning_evaluations"

        # Find the next available directory name for the experiment
        counter = 1
        save_dir = base_dir / f"experiment_{counter}"
        while save_dir.exists():
            counter += 1
            save_dir = base_dir / f"experiment_{counter}"

        save_dir.mkdir(parents=True, exist_ok=True)

        # Save the accuracy plot
        plot_path = save_dir / "plot.png"
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        print(f"Plot saved to {plot_path}.")

        # Save the mean and standard deviation accuracies
        mean_pred_accuracies_df = pd.DataFrame(
            all_mean_accuracies, 
            index=[f"{sampling_method.name}" for sampling_method in self.sampling_methods]
        )
        std_pred_accuracies_df = pd.DataFrame(
            all_std_accuracies, 
            index=[f"{sampling_method.name}" for sampling_method in self.sampling_methods]
        )

        mean_pred_accuracies_csv_path = save_dir / "one_step_pred_accuracies_mean.csv"
        std_pred_accuracies_csv_path = save_dir / "one_step_pred_accuracies_std.csv"

        mean_pred_accuracies_df.to_csv(mean_pred_accuracies_csv_path)
        std_pred_accuracies_df.to_csv(std_pred_accuracies_csv_path)

        print(f"Mean accuracies saved to {mean_pred_accuracies_csv_path}.")
        print(f"Standard deviations saved to {std_pred_accuracies_csv_path}.")

        # Save hyperparameters
        hyperparams = self.params_to_dict()
        hyperparams_path = save_dir / "hyperparameters.json"
        with open(hyperparams_path, "w") as f:
            json.dump(hyperparams, f, indent=3)
        print(f"Hyperparameters saved to {hyperparams_path}.")

        # Save state trajectories
        trajectories_dir = save_dir / "state_trajectories"
        trajectories_dir.mkdir(parents=True, exist_ok=True)

        for sampling_method, repetitions in state_trajectories.items():
            method_dir = trajectories_dir / sampling_method
            method_dir.mkdir(parents=True, exist_ok=True)
            
            for repetition, trajectories in repetitions.items():
                # Save all iterations' trajectories for this repetition into one file
                repetition_file = method_dir / f"repetition_{repetition}.npy"
                np.save(repetition_file, trajectories)

        print(f"State trajectories saved to {trajectories_dir}.")

        # Save iteration-level training results
        training_results_dir = save_dir / "training_results"
        training_results_dir.mkdir(parents=True, exist_ok=True)

        for sampling_method_name, repetition_dict in training_results.items():
            method_dir = training_results_dir / sampling_method_name
            method_dir.mkdir(parents=True, exist_ok=True)

            for repetition, iterations in repetition_dict.items():
                repetition_dir = method_dir / f"repetition_{repetition}"
                repetition_dir.mkdir(parents=True, exist_ok=True)

                for iteration, results_dict in iterations.items():
                    # results_dict contains keys: "train_loss", "test_loss", and "model_weights"
                    iteration_dir = repetition_dir / f"iteration_{iteration+1}"
                    iteration_dir.mkdir(parents=True, exist_ok=True)

                    # Save train and test losses as .npy
                    np.save(iteration_dir / "train_loss.npy", np.array(results_dict["train_loss"]))
                    np.save(iteration_dir / "test_loss.npy", np.array(results_dict["test_loss"]))

                    # Save model weights
                    torch.save(results_dict["model_weights"], iteration_dir / "model_weights.pt")

                    # Plot and save the loss plot for this iteration
                    plt.figure()
                    plt.plot(results_dict["train_loss"], label="Train Loss")
                    if len(results_dict["test_loss"]) > 0:
                        plt.plot(results_dict["test_loss"], label="Test Loss")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.title(f"Train and Test Losses - Avtive Learning Iteration {iteration+1}")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(iteration_dir / "loss_plot.png", bbox_inches="tight", dpi=300)
                    plt.close()
              

    def create_active_learning_plot(
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