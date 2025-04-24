import argparse
from typing import List, Tuple, Dict, Optional
import flwr as fl
from flwr.common import Metrics, Parameters, FitRes, EvaluateRes, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Flower Federated Learning for Object Detection")
parser.add_argument(
    "--server_address",
    type=str,
    default="0.0.0.0:8080",
    help="gRPC server address (default '0.0.0.0:8080')",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=5,
    help="Number of rounds of federated learning (default: 5)",
)
parser.add_argument(
    "--sample_fraction",
    type=float,
    default=1.0,
    help="Fraction of available clients used for fit/evaluate (default: 1.0)",
)
parser.add_argument(
    "--min_num_clients",
    type=int,
    default=2,
    help="Minimum number of available clients required for sampling (default: 2)",
)
parser.add_argument(
    "--min_available_clients",
    type=int,
    default=2,
    help="Minimum number of clients that need to be available before training round can start (default: 2)",
)
parser.add_argument(
    "--save_model_path",
    type=str,
    default="./global_model",
    help="Directory to save the global model after each round (default: ./global_model)",
)

# Define metric aggregation function for object detection
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregates metrics from clients, with emphasis on both accuracy and detection rate."""
    if not metrics:
        return {"accuracy": 0.0, "mean_detections": 0.0}
    
    # Extract metrics
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    detection_rates = [num_examples * m.get("mean_detections", 0.0) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Calculate weighted averages - FIXED: Check if sum of examples is zero
    total_examples = sum(examples)
    
    if total_examples <= 0:
        # If no examples, use simple average instead
        avg_accuracy = sum([m["accuracy"] for _, m in metrics]) / len(metrics) if metrics else 0.0
        avg_detection_rate = sum([m.get("mean_detections", 0.0) for _, m in metrics]) / len(metrics) if metrics else 0.0
    else:
        avg_accuracy = sum(accuracies) / total_examples
        avg_detection_rate = sum(detection_rates) / total_examples
    
    return {
        "accuracy": avg_accuracy,
        "mean_detections": avg_detection_rate,
        "round_participation": len(metrics)
    }

# Configuration for client training
def fit_config(server_round: int) -> Dict[str, any]:
    """Return training configuration for clients."""
    # Object detection clients don't use traditional epochs/batch_size
    # but we include round number for logging purposes
    config = {
        "round": server_round,
    }
    return config

# Configuration for client evaluation
def evaluate_config(server_round: int) -> Dict[str, any]:
    """Return evaluation configuration for clients."""
    # Include round number for logging purposes
    config = {
        "round": server_round,
    }
    return config

# Custom FedAvg with safe aggregation to prevent division by zero
class SafeFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """Aggregate fit results using weighted average, handling empty results."""
        if not results:
            # Return empty aggregation result and metrics
            return None, {"empty_results": True}
        
        # Check if any client has reported examples
        num_examples_total = sum([fit_res.num_examples for _, fit_res in results])
        if num_examples_total == 0:
            print(f"Warning: Round {server_round} - All clients reported zero examples! Using simple average.")
            # Use simple averaging instead of weighted when no examples are reported
            weights = [1.0 / len(results) for _ in results]
        else:
            # Regular weighted averaging
            weights = [fit_res.num_examples / num_examples_total for _, fit_res in results]
        
        # Aggregate parameters and return
        aggregated_parameters = fl.common.aggregate(
            [fit_res.parameters for _, fit_res in results], weights
        )
        
        metrics = {
            "num_clients": len(results),
            "num_examples": num_examples_total,
        }
        
        return aggregated_parameters, metrics

    # FIX: Added protection against division by zero in aggregate_evaluate too
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        """Aggregate evaluation results using weighted average."""
        if not results:
            return None, {}
        
        # Determine if we need to use weighted or simple average
        num_examples_total = sum([eval_res.num_examples for _, eval_res in results])
        
        if num_examples_total == 0:
            print(f"Warning: Round {server_round} - All clients reported zero examples in evaluation! Using simple average.")
            # If no examples, use simple average for loss
            loss = sum([eval_res.loss for _, eval_res in results]) / len(results) if results else None
            
            # Aggregate custom metrics
            metrics_aggregated = {}
            for _, res in results:
                for key, value in res.metrics.items():
                    if key not in metrics_aggregated:
                        metrics_aggregated[key] = []
                    metrics_aggregated[key].append(value)
            
            # Average the metrics
            metrics = {k: sum(v) / len(v) for k, v in metrics_aggregated.items()}
            metrics["num_examples"] = 0
            
            return loss, metrics
        
        # Use the parent's implementation if we have examples
        return super().aggregate_evaluate(server_round, results, failures)

# Custom strategy with model saving capabilities
class SaveModelStrategy(SafeFedAvg):
    def __init__(
        self,
        *args,
        save_path: str = "./global_model",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_path = save_path
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """Aggregate model weights and save the global model after each round."""
        # Aggregate weights using parent class method
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            # Convert parameters to NumPy ndarrays
            aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
            
            # Save aggregated_ndarrays
            import os
            import torch
            import numpy as np
            
            # Create model directory if it doesn't exist
            os.makedirs(self.save_path, exist_ok=True)
            
            # Save the model weights
            filename = f"{self.save_path}/global_model_round_{server_round}.pt"
            
            # Create dummy model to get keys
            try:
                model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
                state_dict = model.state_dict()
                keys = list(state_dict.keys())
                
                # Map aggregated parameters to keys
                updated_state_dict = {}
                for i, key in enumerate(keys):
                    if i < len(aggregated_ndarrays):
                        updated_state_dict[key] = torch.tensor(aggregated_ndarrays[i])
                
                # Save the updated state dict
                torch.save(updated_state_dict, filename)
                print(f"Global model saved at {filename}")
            except Exception as e:
                print(f"Error saving model: {e}")
                # Fallback: save raw parameters
                np.save(f"{self.save_path}/global_model_round_{server_round}_raw.npy", aggregated_ndarrays)
                print(f"Raw parameters saved instead")
            
        return aggregated_parameters, metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        """Aggregate evaluation results and log detailed metrics."""
        if not results:
            return None, {"empty_results": True}
            
        aggregated_loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        
        # Save detailed evaluation metrics
        if results:
            import json
            import os
            
            # Create metrics directory if it doesn't exist
            os.makedirs(f"{self.save_path}/metrics", exist_ok=True)
            
            # Compile all client metrics
            round_metrics = {
                "round": server_round,
                "aggregated_metrics": metrics,
                "client_metrics": []
            }
            
            # Add individual client metrics
            for client_proxy, res in results:
                client_metrics = {
                    "client_id": client_proxy.cid,
                    "loss": res.loss,
                    "metrics": res.metrics
                }
                round_metrics["client_metrics"].append(client_metrics)
            
            # Save to file
            with open(f"{self.save_path}/metrics/round_{server_round}_metrics.json", "w") as f:
                json.dump(round_metrics, f, indent=2)
        
        return aggregated_loss, metrics

def main():
    # Parse arguments
    args = parser.parse_args()
    print(f"Server configuration: {args}")
    
    # Define strategy with model saving
    strategy = SaveModelStrategy(
        fraction_fit=args.sample_fraction,
        fraction_evaluate=args.sample_fraction,
        min_fit_clients=args.min_num_clients,
        min_evaluate_clients=args.min_num_clients,
        min_available_clients=args.min_available_clients,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        evaluate_metrics_aggregation_fn=weighted_average,
        save_path=args.save_model_path,
    )
    
    # Start Flower server
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
