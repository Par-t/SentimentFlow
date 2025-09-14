"""
MLflow Utilities for MLOps Sentiment Analysis Pipeline
"""
import mlflow
import mlflow.sklearn
import mlflow.tracking
from mlflow.tracking import MlflowClient
from mlflow.entities import Experiment, Run
from typing import Dict, Any, Optional, List
import os
import logging
from datetime import datetime
from src.training_config import MLFLOW_CONFIG, DEFAULT_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_mlflow_experiment(experiment_name: str = None, 
                          tracking_uri: str = None,
                          tags: Dict[str, str] = None) -> str:
    """
    Set up MLflow experiment and return experiment ID
    
    Args:
        experiment_name: Name of the experiment
        tracking_uri: MLflow tracking URI
        tags: Additional tags for the experiment
    
    Returns:
        Experiment ID
    """
    try:
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow.set_tracking_uri(MLFLOW_CONFIG['tracking_uri'])
        
        # Get or create experiment
        experiment_name = experiment_name or MLFLOW_CONFIG['experiment_name']
        experiment_id = create_experiment_if_not_exists(experiment_name, tags)
        
        # Set the active experiment
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"MLflow experiment '{experiment_name}' set up successfully")
        logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
        logger.info(f"Experiment ID: {experiment_id}")
        
        return experiment_id
        
    except Exception as e:
        logger.error(f"Error setting up MLflow experiment: {e}")
        raise


def create_experiment_if_not_exists(name: str, 
                                  tags: Dict[str, str] = None) -> str:
    """
    Create experiment if it doesn't exist, return experiment ID
    
    Args:
        name: Experiment name
        tags: Tags for the experiment
    
    Returns:
        Experiment ID
    """
    try:
        client = get_mlflow_client()
        
        # Check if experiment exists
        experiment = client.get_experiment_by_name(name)
        
        if experiment is None:
            # Create new experiment
            experiment_id = client.create_experiment(
                name=name,
                tags=tags or MLFLOW_CONFIG['default_tags']
            )
            logger.info(f"Created new experiment: {name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {name} (ID: {experiment_id})")
        
        return experiment_id
        
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        raise


def get_mlflow_client() -> MlflowClient:
    """
    Get MLflow client instance
    
    Returns:
        MLflowClient instance
    """
    try:
        client = MlflowClient()
        return client
    except Exception as e:
        logger.error(f"Error creating MLflow client: {e}")
        raise


def log_experiment_info(experiment_name: str, 
                       description: str = None,
                       tags: Dict[str, str] = None) -> None:
    """
    Log experiment information and metadata
    
    Args:
        experiment_name: Name of the experiment
        description: Experiment description
        tags: Additional tags
    """
    try:
        with mlflow.start_run(run_name=f"experiment_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log experiment metadata
            mlflow.log_param("experiment_name", experiment_name)
            mlflow.log_param("description", description or "Sentiment Analysis ML Pipeline")
            mlflow.log_param("created_at", datetime.now().isoformat())
            mlflow.log_param("python_version", mlflow.__version__)
            
            # Log default tags
            default_tags = MLFLOW_CONFIG['default_tags'].copy()
            if tags:
                default_tags.update(tags)
            
            for key, value in default_tags.items():
                mlflow.log_param(f"tag_{key}", value)
            
            logger.info(f"Logged experiment info for: {experiment_name}")
            
    except Exception as e:
        logger.error(f"Error logging experiment info: {e}")
        raise


def start_mlflow_run(run_name: str = None, 
                    tags: Dict[str, str] = None) -> mlflow.ActiveRun:
    """
    Start a new MLflow run
    
    Args:
        run_name: Name for the run
        tags: Tags for the run
    
    Returns:
        Active MLflow run
    """
    try:
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        run = mlflow.start_run(run_name=run_name)
        
        # Log run metadata
        mlflow.log_param("run_name", run_name)
        mlflow.log_param("start_time", datetime.now().isoformat())
        
        # Log tags
        if tags:
            for key, value in tags.items():
                mlflow.log_param(f"tag_{key}", value)
        
        logger.info(f"Started MLflow run: {run_name}")
        return run
        
    except Exception as e:
        logger.error(f"Error starting MLflow run: {e}")
        raise


def log_model_parameters(model_name: str, 
                        parameters: Dict[str, Any]) -> None:
    """
    Log model parameters to MLflow
    
    Args:
        model_name: Name of the model
        parameters: Model parameters to log
    """
    try:
        for key, value in parameters.items():
            mlflow.log_param(f"{model_name}_{key}", value)
        
        logger.info(f"Logged parameters for model: {model_name}")
        
    except Exception as e:
        logger.error(f"Error logging model parameters: {e}")
        raise


def log_model_metrics(metrics: Dict[str, float]) -> None:
    """
    Log model metrics to MLflow
    
    Args:
        metrics: Dictionary of metrics to log
    """
    try:
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        logger.info(f"Logged {len(metrics)} metrics to MLflow")
        
    except Exception as e:
        logger.error(f"Error logging metrics: {e}")
        raise


def log_model_artifacts(artifacts: Dict[str, str]) -> None:
    """
    Log model artifacts to MLflow
    
    Args:
        artifacts: Dictionary of artifact_name -> file_path
    """
    try:
        for artifact_name, file_path in artifacts.items():
            if os.path.exists(file_path):
                mlflow.log_artifact(file_path, artifact_name)
                logger.info(f"Logged artifact: {artifact_name}")
            else:
                logger.warning(f"Artifact file not found: {file_path}")
        
    except Exception as e:
        logger.error(f"Error logging artifacts: {e}")
        raise


def log_model(model, 
              model_name: str, 
              model_type: str = "sklearn",
              registered_model_name: str = None) -> None:
    """
    Log model to MLflow
    
    Args:
        model: Trained model object
        model_name: Name for the model
        model_type: Type of model (sklearn, pytorch, etc.)
        registered_model_name: Name for model registry
    """
    try:
        if model_type == "sklearn":
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=model_name,
                registered_model_name=registered_model_name
            )
        else:
            # For other model types, use generic log_model
            mlflow.log_model(
                model=model,
                artifact_path=model_name,
                registered_model_name=registered_model_name
            )
        
        logger.info(f"Logged model: {model_name}")
        
    except Exception as e:
        logger.error(f"Error logging model: {e}")
        raise


def get_experiment_runs(experiment_name: str = None) -> List[Run]:
    """
    Get all runs for an experiment
    
    Args:
        experiment_name: Name of the experiment
    
    Returns:
        List of runs
    """
    try:
        client = get_mlflow_client()
        
        if experiment_name:
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                logger.warning(f"Experiment not found: {experiment_name}")
                return []
            experiment_id = experiment.experiment_id
        else:
            experiment_id = mlflow.active_run().info.experiment_id
        
        runs = client.search_runs(experiment_ids=[experiment_id])
        logger.info(f"Found {len(runs)} runs in experiment")
        
        return runs
        
    except Exception as e:
        logger.error(f"Error getting experiment runs: {e}")
        return []


def compare_models(experiment_name: str = None, 
                  metric_name: str = "f1") -> Dict[str, Any]:
    """
    Compare models in an experiment
    
    Args:
        experiment_name: Name of the experiment
        metric_name: Metric to use for comparison
    
    Returns:
        Dictionary with model comparison results
    """
    try:
        runs = get_experiment_runs(experiment_name)
        
        if not runs:
            logger.warning("No runs found for comparison")
            return {}
        
        # Extract model information
        model_comparison = {}
        
        for run in runs:
            run_id = run.info.run_id
            run_name = run.data.tags.get('mlflow.runName', run_id)
            
            # Get metrics
            metrics = run.data.metrics
            metric_value = metrics.get(metric_name, 0.0)
            
            # Get parameters
            params = run.data.params
            
            model_comparison[run_name] = {
                'run_id': run_id,
                'metric_value': metric_value,
                'parameters': params,
                'metrics': metrics
            }
        
        # Sort by metric value
        sorted_models = sorted(
            model_comparison.items(), 
            key=lambda x: x[1]['metric_value'], 
            reverse=True
        )
        
        logger.info(f"Compared {len(sorted_models)} models by {metric_name}")
        
        return {
            'sorted_models': sorted_models,
            'best_model': sorted_models[0] if sorted_models else None,
            'metric_used': metric_name
        }
        
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        return {}


def register_model_to_registry(model_name: str, 
                             model_version: str = "1",
                             stage: str = "Staging") -> str:
    """
    Register model to MLflow model registry
    
    Args:
        model_name: Name of the model
        model_version: Version of the model
        stage: Stage (Staging, Production, Archived)
    
    Returns:
        Registered model URI
    """
    try:
        client = get_mlflow_client()
        
        # Get the latest run
        experiment = client.get_experiment_by_name(MLFLOW_CONFIG['experiment_name'])
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        
        if not runs:
            raise ValueError("No runs found to register model")
        
        latest_run = runs[0]  # Assuming first run is latest
        
        # Register model
        model_uri = f"runs:/{latest_run.info.run_id}/{model_name}"
        registered_model = client.create_registered_model(
            name=MLFLOW_CONFIG['model_registry_name']
        )
        
        model_version = client.create_model_version(
            name=MLFLOW_CONFIG['model_registry_name'],
            source=model_uri,
            run_id=latest_run.info.run_id
        )
        
        # Set stage
        client.transition_model_version_stage(
            name=MLFLOW_CONFIG['model_registry_name'],
            version=model_version.version,
            stage=stage
        )
        
        logger.info(f"Registered model: {model_name} (Version: {model_version.version})")
        return model_uri
        
    except Exception as e:
        logger.error(f"Error registering model: {e}")
        raise


def get_model_from_registry(model_name: str, 
                          stage: str = "Production") -> Any:
    """
    Get model from MLflow model registry
    
    Args:
        model_name: Name of the model
        stage: Stage of the model
    
    Returns:
        Loaded model
    """
    try:
        client = get_mlflow_client()
        
        # Get latest version of the model in the specified stage
        model_versions = client.get_latest_versions(
            name=model_name,
            stages=[stage]
        )
        
        if not model_versions:
            raise ValueError(f"No model found in stage: {stage}")
        
        latest_version = model_versions[0]
        model_uri = f"models:/{model_name}/{latest_version.version}"
        
        # Load model
        model = mlflow.sklearn.load_model(model_uri)
        
        logger.info(f"Loaded model: {model_name} (Version: {latest_version.version})")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model from registry: {e}")
        raise


def end_mlflow_run() -> None:
    """
    End the current MLflow run
    """
    try:
        mlflow.end_run()
        logger.info("Ended MLflow run")
    except Exception as e:
        logger.error(f"Error ending MLflow run: {e}")


def cleanup_old_runs(experiment_name: str, 
                    keep_last_n: int = 10) -> None:
    """
    Clean up old runs, keeping only the last N runs
    
    Args:
        experiment_name: Name of the experiment
        keep_last_n: Number of recent runs to keep
    """
    try:
        client = get_mlflow_client()
        runs = get_experiment_runs(experiment_name)
        
        if len(runs) <= keep_last_n:
            logger.info(f"Only {len(runs)} runs found, no cleanup needed")
            return
        
        # Sort runs by start time (newest first)
        sorted_runs = sorted(runs, key=lambda x: x.info.start_time, reverse=True)
        
        # Delete old runs
        runs_to_delete = sorted_runs[keep_last_n:]
        
        for run in runs_to_delete:
            client.delete_run(run.info.run_id)
            logger.info(f"Deleted old run: {run.info.run_id}")
        
        logger.info(f"Cleaned up {len(runs_to_delete)} old runs")
        
    except Exception as e:
        logger.error(f"Error cleaning up old runs: {e}")


# Convenience function for quick setup
def setup_mlflow_for_training(experiment_name: str = None) -> str:
    """
    Quick setup function for training pipeline
    
    Args:
        experiment_name: Name of the experiment
    
    Returns:
        Experiment ID
    """
    experiment_name = experiment_name or MLFLOW_CONFIG['experiment_name']
    
    # Setup experiment
    experiment_id = setup_mlflow_experiment(experiment_name)
    
    # Log experiment info
    log_experiment_info(
        experiment_name=experiment_name,
        description="Sentiment Analysis Model Training Pipeline",
        tags={
            "pipeline_stage": "training",
            "model_type": "sentiment_classification"
        }
    )
    
    return experiment_id
