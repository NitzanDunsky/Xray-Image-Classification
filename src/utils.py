import logging
import os
import keras

class EpochLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logging.info(f"Epoch {epoch + 1} completed. "
                     f"Loss: {logs.get('loss')}, Accuracy: {logs.get('accuracy')}, "
                     f"Val Loss: {logs.get('val_loss')}, Val Accuracy: {logs.get('val_accuracy')}")


def log_model_summary(model):
    """Helper function to log the model summary without special characters."""
    log_file_path = os.path.join('../logs', 'model_summary.log')

    with open(log_file_path, 'w', encoding='utf-8') as f:
        # Custom print function to format the summary without special characters
        def custom_print_fn(line):
            # Replace special characters with a simpler version (or remove them)
            clean_line = line.replace('├', '|').replace('└', '|').replace('─', '-').replace('┌', '|')
            f.write(clean_line + '\n')

        # Redirect the model summary to custom print function
        model.summary(print_fn=custom_print_fn)

    # Also log that the model summary was saved to the log file
    logging.info(f"Model summary saved to {log_file_path}")