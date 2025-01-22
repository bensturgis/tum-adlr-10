import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from train import train_model
from metrics.evaluation_metric import EvaluationMetric
from sampling_methods.sampling_method import SamplingMethod
from sampling_methods.random_exploration import RandomExploration
from utils.train_utils import combine_datasets, create_dataloader, create_test_dataset
from typing import Dict, List
from pathlib import Path
import json
import torch

class ActiveLearningEvaluator():
    def __init__(
            self, true_env: gym.Env, learned_env: gym.Env, sampling_methods: List[SamplingMethod],
            evaluation_metrics: List[EvaluationMetric], num_al_iterations: int, num_epochs: int, batch_size: int,
            learning_rate: float, num_eval_repetitions: int = 20
        ) -> None:
        """
        Initializes the ActiveLearningEvaluator with the necessary configurations.

        Args:
            true_env (gym.Env): The true environment for data collection.
            learned_env (gym.Env): The learned environment using a dynamics model.
            sampling_methods (List[SamplingMethod]): List of methods to collect (state, action, next state)
                                                     pairs.
            evaluation_metrics (List[EvluationMetric]): List of evaluation metrics.
            num_al_iterations (N_{AL}) (int): Number of active learning iterations.
            num_epochs (int): Number of epochs for training the dynamics model.
            batch_size (int): Batch size for the training dataloader.
            learning_rate (float): Learning rate for the optimizer.
            num_eval_repetitions (int): Number of evaluation repetitions to compute robust metrics.
        """    
        self.true_env = true_env
        self.learned_env = learned_env
        self.sampling_methods = sampling_methods
        self.evaluation_metrics = evaluation_metrics
        self.num_al_iterations = num_al_iterations
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_eval_repetitions = num_eval_repetitions


    def active_learning(self, show: bool = True, save: bool = True) -> None:
        """
        Implements Algorithm 1 from the paper: Actively learning dynamical systems 
        using Bayesian neural networks.
        
        - Collects initial data.
        - Iteratively trains the dynamics model and evaluates its accuracy.
        - Updates the dataset with new trajectories after each iteration.
        - Plots predictive accuracy over iterations.
        """
        # Dictionary to store the accuracy results for each sampling method and metric across all repetitions.
        # all_accuracies[sampling_method_name][metric_name] -> np.ndarray of shape:
        # (num_eval_repetitions, num_al_iterations)
        all_accuracies = {}
        for sampling_method in self.sampling_methods:
            all_accuracies[sampling_method.name] = {}
            for metric in self.evaluation_metrics:
                all_accuracies[sampling_method.name][metric.name] = np.zeros(
                    (self.num_eval_repetitions, self.num_al_iterations)
                )

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
        action_dim = self.true_env.action_space.shape[0]

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

        for sampling_method in self.sampling_methods:
            print("--------------------------------------------------------------------------------------")
            print(f"Starting active learning with the sampling method: '{sampling_method.name}'.")
            print("--------------------------------------------------------------------------------------")

            # Initialize structures for state trajectories and training results for this method
            state_trajectories[sampling_method.name] = {}
            training_results[sampling_method.name] = {}
            
            for repetition in range(self.num_eval_repetitions):
                # Collect initial dataset using random exploration
                random_exploration = RandomExploration(sampling_method.horizon)
                total_dataset = random_exploration.sample(self.true_env)

                # Initialize training results for this repetition
                training_results[sampling_method.name][repetition] = {}

                # Initialize the numpy array for saving state trajectories for all iterations in the repetition
                # Shape: (num_al_iterations, horizon, state_dim + action_dim + state_dim)
                state_trajectories[sampling_method.name][repetition] = np.zeros(
                    (self.num_al_iterations, sampling_method.horizon, state_dim + action_dim + state_dim)
                )
                state_trajectory = torch.cat(total_dataset.tensors[:], dim=1).numpy()  # shape: (horizon, state_dim)
                state_trajectories[sampling_method.name][repetition][0, :, :] = state_trajectory

                # Perform active learning iterations
                for iteration in range(self.num_al_iterations):
                    print(f"Evaluation Repetition {repetition + 1}/{self.num_eval_repetitions}\n"
                          f"Active Learning Iteration {iteration + 1}/{self.num_al_iterations}\n"
                          f"Number of samples: {len(total_dataset)}")
                    
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
                    
                    # Evaluate the learned model 
                    for metric in self.evaluation_metrics:
                        accuracy = metric.evaluate()
                        print(f"{metric.name}: {accuracy}")
                    
                        # Store the accuracy for this repetition and iteration
                        all_accuracies[sampling_method.name][metric.name][repetition][iteration] = accuracy
                    
                    # Collect new data only if further training iterations remain
                    if iteration < self.num_al_iterations - 1:
                        # Sample new data using the specified sampling method
                        new_dataset = sampling_method.sample(self.true_env, self.learned_env)
                        
                        # Merge the new dataset with the existing dataset
                        total_dataset = combine_datasets(total_dataset, new_dataset)
            
                        # Store the newly sampled state trajectory
                        state_trajectory = torch.cat(new_dataset.tensors[:], dim=1).numpy()  # shape: (horizon, state_dim)
                        state_trajectories[sampling_method.name][repetition][iteration + 1, :, :] = state_trajectory

        mean_accuracies = {}
        std_accuracies = {}
        for sampling_method in self.sampling_methods:
            mean_accuracies[sampling_method.name] = {}
            std_accuracies[sampling_method.name] = {}
            for metric in self.evaluation_metrics:                
                # Compute the mean (and std) across the repetition axis = 0
                mean_accuracies[sampling_method.name][metric.name] = list(np.mean(
                    all_accuracies[sampling_method.name][metric.name], axis=0
                ))

                std_accuracies[sampling_method.name][metric.name] = list(np.std(
                    all_accuracies[sampling_method.name][metric.name], axis=0
                ))

        # Plot and save the results
        if show or save:
            figures = self.create_active_learning_plot(mean_accuracies=mean_accuracies,
                                                       std_accuracies=std_accuracies)
        
        if save:
            self.save_active_learning_results(figures=figures,
                                              mean_accuracies=mean_accuracies,
                                              std_accuracies=std_accuracies,
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
            "metrics": [
                metric.params_to_dict() for metric in self.evaluation_metrics
            ],
        }
        return parameter_dict


    def save_active_learning_results(
            self, figures: Dict[str, Figure],
            mean_accuracies: Dict[str, Dict[str, List[float]]],
            std_accuracies: Dict[str, Dict[str, List[float]]],
            state_trajectories: Dict[str, Dict[int, np.ndarray]],
            training_results: Dict[str, Dict[int, List[Dict]]]
    ) -> None:
        """
        Saves the results of the active learning experiment, including:
        - A summary plot of predictive accuracies over iterations for each metric.
        - Mean and standard deviation of predictive accuracies for each sampling method.
        - Hyperparameters of the experiment.
        - State trajectories collected during the experiment.
        - Iteration-level training results (train/test losses, model weights, and a loss plot).

        Args:
            figures (Dict[str, Figure]): A dictionary where each key is a metric name and each
                value is a Matplotlib figure object. These figures will be saved as PNG files.
            mean_accuracies (Dict[str, Dict[str, List]]): A nested dictionary where
                mean_accuracies[sampling_method][metric] is a list of length
                `num_al_iterations` with mean accuracies per iteration across repetitions.
            std_accuracies (Dict[str, Dict[str, List]]): A nested dictionary where
                std_accuracies[sampling_method][metric] is a list of length 
                `num_al_iterations` with standard deviations per iteration across repetitions.
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

        # Save the accuracy plots
        for metric in self.evaluation_metrics:
            plot_path = save_dir / f"{metric.name.lower().replace(' ', '_')}_plot.png"
            figures[metric.name].savefig(plot_path, bbox_inches="tight", dpi=300)
            print(f"{metric.name} plot saved to {plot_path}.")

        # Save the mean and standard deviation of the predictive accuracies
        mean_pred_accuracies_path = save_dir / "pred_accuracies_mean.json"
        with open(mean_pred_accuracies_path, "w") as f:
            json.dump(mean_accuracies, f, indent=3)
        print(f"Mean accuracies saved to {mean_pred_accuracies_path}.")
        
        std_pred_accuracies_path = save_dir / "pred_accuracies_std.json"
        with open(std_pred_accuracies_path, "w") as f:
            json.dump(std_accuracies, f, indent=3)        
        print(f"Standard deviations saved to {std_pred_accuracies_path}.")

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
                    plt.title(f"Train and Test Losses - Active Learning Iteration {iteration+1}")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(iteration_dir / "loss_plot.png", bbox_inches="tight", dpi=300)
                    plt.close()
              

    def create_active_learning_plot(
        self, mean_accuracies: Dict[str, Dict[str, List[float]]],
        std_accuracies: Dict[str, Dict[str, List[float]]]
    ) -> None:
        """
        Plots the mean and standard deviation of predictive accuracies for different sampling
        methods over active learning iterations.

        Args:
            mean_accuracies (Dict[str, Dict[str, List]]): A nested dictionary where
                mean_accuracies[sampling_method][metric] is a list of length
                `num_al_iterations` with mean accuracies per iteration across repetitions.
            std_accuracies (Dict[str, Dict[str, List]]): A nested dictionary where
                std_accuracies[sampling_method][metric] is a list of length 
                `num_al_iterations` with standard deviations per iteration across repetitions.
        """
        # Define map of colors to distinguish the sampling methods
        color_map = {
            "Random Exploration": "blue",
            "Random Sampling Shooting": "green",
        }
        
        iterations = range(1, self.num_al_iterations + 1)

        # Create a plot for each evaluation metric
        figures = {}
        for metric in self.evaluation_metrics:
            fig, ax = plt.subplots(
                num=f"{metric.name} over Active Learning Iterations", figsize=(8, 5)
            )
            
            # Plot mean and std for each sampling method using the specified color
            for sampling_method in self.sampling_methods:
                # Extract arrays without overwriting the original dictionary variables
                mean_arr = mean_accuracies[sampling_method.name][metric.name]
                std_arr = std_accuracies[sampling_method.name][metric.name]

                # Lookup the color for this method from the map
                color = color_map.get(sampling_method.name, "black")  # default to black if not found

                # Plot the mean curve
                ax.plot(iterations, mean_arr, label=sampling_method.name, color=color)
                # Plot error bars (±1 std)
                ax.errorbar(
                    iterations, mean_arr,
                    yerr=std_arr, fmt='o',
                    color=color, ecolor=color,
                    capsize=3, elinewidth=1
                )

            # Customize axes and title
            ax.set_xlabel("Active Learning Iteration")
            ax.set_ylabel(metric.name)
            ax.set_title(f"{metric.name} over Active Learning Iterations")
            ax.legend()
            ax.grid(True)

            figures[metric.name] = fig

        return figures