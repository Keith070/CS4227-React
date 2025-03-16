import logging
from ClassifierFacade import ClassifierFacade
from sklearn.model_selection import train_test_split
from config_manager import ConfigurationManager  # Import the ConfigurationManager
from observer import Subject, LoggingObserver  # Import the Observer classes

# Initialize the ConfigurationManager and load the config
config = ConfigurationManager()  # Access the singleton instance
config.load_configuration("config.json")  # Load configuration from the JSON file

# Setting up logging using the configuration manager
logging.basicConfig(
    filename=config.logging_file,  # Use the correct config value for log file
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_email_classification_system():
    facade = ClassifierFacade()  # Create an instance of the facade

    # Initialize the subject and logging observer
    subject = Subject()
    logging_observer = LoggingObserver(log_file=config.logging_file)
    subject.attach(logging_observer)

    while True:
        # Load dataset using configuration manager for dataset path
        print("Please select a dataset to load:")
        print("1. Gallery App Dataset")
        print("2. Purchasing Emails Dataset")
        dataset_choice = input("Enter choice: ")

        # Log the dataset choice
        logging.info(f"Loading dataset: {dataset_choice}")

        try:
            df = facade.load_dataset(dataset_choice)
            if df is None:
                raise ValueError("Failed to load the dataset.")
        except ValueError as e:
            print(f"Error: {e}")
            continue

        # Log the start of preprocessing
        logging.info("Preprocessing data...")
        df = facade.preprocess_data(df)
        logging.info("Preprocessing complete. Email body cleaned.")

        # Log the start of vectorization
        logging.info("Vectorizing data...")
        X, y = facade.vectorize_data(df)
        logging.info("Vectorization complete.")

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        while True:
            print("\nSelect classification model:")
            print("1. Random Forest")
            print("2. SVM")
            print("3. AdaBoost")
            print("4. Naive Bayes")
            print("5. Logistic Regression")
            model_choice = input("Enter choice: ")

            classifier_map = {
                '1': 'random_forest',
                '2': 'svm',
                '3': 'adaboost',
                '4': 'naive_bayes',
                '5': 'logistic_regression'
            }
            classifier_choice = classifier_map.get(model_choice)
            if not classifier_choice:
                print("Invalid choice, please try again.")
                continue

            # Log classifier selection
            logging.info(f"Choosing {classifier_choice} classifier with hyperparameter tuning.")
            if classifier_choice == 'svm':
                logging.info("Performing grid search for hyperparameter tuning.")

            # Choose and train the model
            try:
                best_model = facade.choose_classifier(classifier_choice, X_train, y_train)
                accuracy, conf_matrix = facade.evaluate_model(best_model, X_test, y_test)
            except Exception as e:
                print(f"Error in model training or evaluation: {e}")
                logging.error(f"Error in model training or evaluation: {e}")
                continue

            # Log the results
            logging.info(f"Best {classifier_choice} Classifier Accuracy: {accuracy}")
            logging.info(f"Best Parameters: {best_model.get_params()}")

            # Log model evaluation
            logging.info("Evaluating model...")
            logging.info(f"Accuracy: {accuracy * 100:.2f}%")
            logging.info(f"Confusion Matrix:\n{conf_matrix}")

            # Log the results through the observer
            subject.notify({
                "accuracy": f"{accuracy * 100:.2f}%",
                "confusion_matrix": conf_matrix.tolist()
            })

            print(f"\nAccuracy: {accuracy * 100:.2f}%")
            print("Confusion Matrix:")
            print(conf_matrix)

            print("\nWould you like to:")
            print("1. Test with another model")
            print("2. Load a new dataset")
            print("3. Exit")
            next_step = input("Enter choice: ")
            if next_step == '1':
                continue
            elif next_step == '2':
                break
            elif next_step == '3':
                logging.info("Exiting email classification system.")
                print("Exiting email classification system complete.")
                return
            else:
                print("Invalid choice, exiting.")
                return

if __name__ == "__main__":
    run_email_classification_system()
