import matplotlib.pyplot as plt
from train import create_dataloader, train_model
from sampling_methods.random_exploration import random_exploration
from utils import combine_datasets
from metrics.one_step_pred_accuracy import compute_one_step_pred_accuracy


class ActiveLearningEvaluator():
    def __init__(self, true_env, learned_env, horizon, num_al_iterations,
                 num_epochs, batch_size, learning_rate):
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

        Note:
            This class implements Algorithm 1 from the referenced paper.
        """    
        self.horizon = horizon
        self.num_iterations = num_al_iterations
        self.true_env = true_env
        self.learned_env = learned_env
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def active_learning(self):
        """
        Implements Algorithm 1 from the paper: Actively learning dynamical systems 
        using Bayesian neural networks.
        
        - Collects initial data using random exploration.
        - Iteratively trains the dynamics model and evaluates its accuracy.
        - Updates the dataset with new trajectories after each iteration.
        - Plots the one-step predictive accuracy over iterations.
        """
        # Collect initial dataset using random exploration
        total_dataset = random_exploration(self.true_env, self.horizon)
        
        # Initialize a list to store one-step predictive accuracy for each iteration
        accuracy_history = []

        # Perform active learning iterations
        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration + 1}/{self.num_iterations}")
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
        
        # Plot the one-step predictive accuracy over iterations
        plt.figure("One Step Predictive Accuracy", figsize=(8, 6))
        plt.plot(range(1, self.num_iterations + 1), accuracy_history, marker='o', label="One-step Predictive Accuracy")
        plt.xlabel("Active Learning Iteration")
        plt.ylabel("One-step Predictive Accuracy")
        plt.title("One-step Predictive Accuracy Over Active Learning Iterations (Algorithm 1)")
        plt.grid(True)
        plt.legend()
        plt.show()