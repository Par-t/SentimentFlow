"""
Simple Experiment Tracker

A lightweight tracking system that maintains a JSON file mapping datasets to their artifacts.
Just keeps track of what files were generated for each dataset.
"""

import json
import boto3
from datetime import datetime
from typing import Dict, List

class SimpleTracker:
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
        self.tracking_file = "experiment_tracking.json"
        
    def load_tracking(self) -> Dict:
        """Load tracking data from S3"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=f"tracking/{self.tracking_file}"
            )
            return json.loads(response['Body'].read().decode('utf-8'))
        except:
            # Return empty structure if file doesn't exist
            return {}
    
    def save_tracking(self, data: Dict) -> bool:
        """Save tracking data to S3"""
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=f"tracking/{self.tracking_file}",
                Body=json.dumps(data, indent=2),
                ContentType='application/json'
            )
            return True
        except Exception as e:
            print(f"Error saving tracking data: {e}")
            return False
    
    def add_artifact(self, dataset_name: str, artifact_type: str, filename: str):
        """Add an artifact to the tracking"""
        tracking = self.load_tracking()
        
        # Initialize dataset entry if it doesn't exist
        if dataset_name not in tracking:
            tracking[dataset_name] = {}
        
        # Initialize exp_id counter if it doesn't exist
        if 'exp_counter' not in tracking[dataset_name]:
            tracking[dataset_name]['exp_counter'] = 0
        
        # Get current experiment ID
        exp_id = tracking[dataset_name]['exp_counter']
        
        # Check if we need to create a new experiment by comparing parameters
        if str(exp_id) in tracking[dataset_name]:
            if self._should_create_new_experiment(tracking[dataset_name], str(exp_id)):
                # Increment counter for new experiment
                tracking[dataset_name]['exp_counter'] += 1
                exp_id = tracking[dataset_name]['exp_counter']
                print(f"🆕 Creating new experiment {exp_id} due to parameter changes")
        
        # Initialize experiment entry if it doesn't exist
        if str(exp_id) not in tracking[dataset_name]:
            tracking[dataset_name][str(exp_id)] = {
                'created_at': datetime.now().isoformat(),
                'processed_data_file': None,
                'train_split': None,
                'test_split': None,
                'val_split': None,
                'train_features': None,
                'test_features': None,
                'val_features': None,
                'feature_extractor': None,
                'label_mapping': None,
                'split_params': None,
                'feature_params': None
            }
        
        # Add the artifact
        tracking[dataset_name][str(exp_id)][artifact_type] = filename
        
        # Save back to S3
        self.save_tracking(tracking)
        print(f"📝 Tracked {artifact_type}: {filename}")
    
    def _should_create_new_experiment(self, dataset_tracking: dict, current_exp_id: str) -> bool:
        """Check if current parameters differ from the last experiment"""
        try:
            # Get the last experiment's parameters
            last_exp = dataset_tracking[current_exp_id]
            
            # Load current config parameters
            from config import TRAIN_SIZE, TEST_SIZE, VAL_SIZE, RANDOM_STATE
            
            # Check if split_params exist in the last experiment
            if last_exp.get('split_params'):
                last_split_params_str = last_exp['split_params']
                last_split_params = json.loads(last_split_params_str)
                
                # Compare with current config values
                if (last_split_params.get('train_size') != TRAIN_SIZE or
                    last_split_params.get('test_size') != TEST_SIZE or
                    last_split_params.get('val_size') != VAL_SIZE or
                    last_split_params.get('random_state') != RANDOM_STATE):
                    
                    print(f"🔄 Split parameters changed:")
                    print(f"  Last: {last_split_params}")
                    print(f"  Current: train_size={TRAIN_SIZE}, test_size={TEST_SIZE}, val_size={VAL_SIZE}, random_state={RANDOM_STATE}")
                    return True
            
            # Check feature_params if they exist
            if last_exp.get('feature_params'):
                # For now, we'll assume feature params haven't changed
                # This could be enhanced to compare feature extraction config
                pass
            
            # If we get here, parameters are the same
            return False
            
        except Exception as e:
            print(f"⚠️ Error comparing parameters: {e}")
            # If there's an error comparing, assume new experiment
            return True
    
    def get_dataset_experiments(self, dataset_name: str) -> Dict:
        """Get all experiments for a dataset"""
        tracking = self.load_tracking()
        return tracking.get(dataset_name, {})
    
    def list_datasets(self) -> List[str]:
        """List all tracked datasets"""
        tracking = self.load_tracking()
        return list(tracking.keys())
    
    def get_split_params(self, dataset_name: str, exp_id: str = "0") -> Dict:
        """Get split parameters for reproducing the same train/test/val split"""
        tracking = self.load_tracking()
        try:
            split_params_str = tracking[dataset_name][exp_id].get('split_params')
            if split_params_str:
                return json.loads(split_params_str)
            return {}
        except:
            return {}
    
    def get_feature_params(self, dataset_name: str, exp_id: str = "0") -> Dict:
        """Get feature extraction parameters for reproducing the same features"""
        tracking = self.load_tracking()
        try:
            feature_params_str = tracking[dataset_name][exp_id].get('feature_params')
            if feature_params_str:
                return json.loads(feature_params_str)
            return {}
        except:
            return {}