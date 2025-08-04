import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigLoader:
    """
    Configuration loader for the AnimeOrNot application.
    Loads settings from config.yaml file.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dictionary containing configuration
        """
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_path} not found. Using defaults.")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration if file is not found.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "app": {
                "name": "AnimeOrNot API",
                "version": "1.0.0",
                "host": "0.0.0.0",
                "port": 8000
            },
            "model": {
                "name": "ResNet50",
                "num_classes": 2,
                "input_size": [224, 224],
                "model_path": "best_model.pth",
                "class_names": ["Anime", "Cartoon"]
            },
            "device": {
                "type": "auto"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "app.host")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model-specific configuration.
        
        Returns:
            Model configuration dictionary
        """
        return self.config.get("model", {})
    
    def get_app_config(self) -> Dict[str, Any]:
        """
        Get application-specific configuration.
        
        Returns:
            Application configuration dictionary
        """
        return self.config.get("app", {})
    
    def get_device_config(self) -> Dict[str, Any]:
        """
        Get device-specific configuration.
        
        Returns:
            Device configuration dictionary
        """
        return self.config.get("device", {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """
        Get data processing configuration.
        
        Returns:
            Data configuration dictionary
        """
        return self.config.get("data", {})
    
    def get_api_config(self) -> Dict[str, Any]:
        """
        Get API-specific configuration.
        
        Returns:
            API configuration dictionary
        """
        return self.config.get("api", {})
    
    def reload(self) -> None:
        """
        Reload configuration from file.
        """
        self.config = self._load_config()

# Global configuration instance
config = ConfigLoader() 