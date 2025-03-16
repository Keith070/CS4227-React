import json
import threading

class ConfigurationManager:
    """
    Singleton class to manage configuration settings for the application.
    Ensures that only one instance of the configuration is created and provides thread-safe access.
    """
    _instance = None
    _lock = threading.Lock()  # Lock to ensure thread-safety during initialization

    def __new__(cls):
        """Create and initialize the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ConfigurationManager, cls).__new__(cls)
                    # Initialize shared resources lazily
                    cls._instance._config = {}  # Empty dictionary to hold configurations
        return cls._instance

    def load_configuration(self, config_file="config.json"):
        """Load configuration from the provided JSON file."""
        try:
            with open(config_file, 'r') as f:
                self._config = json.load(f)  # Load the configuration data from the file
            print(f"Configuration loaded from {config_file}")
        except Exception as e:
            print(f"Error loading configuration: {e}")
    
    def get_configuration(self, key):
        """Retrieve a configuration value by its key."""
        return self._config.get(key, None)

    def set_configuration(self, key, value):
        """Set a configuration value."""
        self._config[key] = value
        print(f"Configuration key '{key}' set to '{value}'")
    
    # Optionally, you can add default fallback methods as before
    @property
    def dataset_path(self):
        """Get the dataset path, falling back to default if not found."""
        return self.get_configuration('dataset_path') or 'data/'

    @property
    def logging_file(self):
        """Get the logging file path, falling back to default if not found."""
        return self.get_configuration('logging_file') or 'email_classification.log'
