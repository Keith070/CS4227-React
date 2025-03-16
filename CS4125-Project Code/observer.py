class Observer:
    def update(self, data):
        """Receive update from subject."""
        pass


class Subject:
    def __init__(self):
        self._observers = []

    def attach(self, observer):
        """Attach an observer."""
        self._observers.append(observer)

    def detach(self, observer):
        """Detach an observer."""
        self._observers.remove(observer)

    def notify(self, data):
        """Notify all observers."""
        for observer in self._observers:
            observer.update(data)


class LoggingObserver(Observer):
    def __init__(self, log_file):
        """Initialize with the log file path."""
        self.log_file = log_file
        self.previous_data = None  # Track the last data to avoid duplicate logs

    def update(self, data):
        """Log classification updates to the specified log file."""
        if self.previous_data != data['accuracy']:  # Only log if data is different
            try:
                # Write to log file
                with open(self.log_file, 'a') as f:
                    f.write(f"LoggingObserver: Accuracy: {data['accuracy']}\n")

                # Update the previous data to avoid duplicate logs
                self.previous_data = data['accuracy']

            except Exception as e:
                print(f"Error logging data: {e}")
