from kfp.v2.dsl import component, Input, Output, Dataset, Model, Artifact


@component(base_image="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/tf2-cpu.2-17.py310")
def evaluate_model(processed_test_data: Input[Dataset],
                   test_labels_data: Input[Dataset],
                   model: Input[Model],
                   evaluation_metrics: Output[Artifact]):
    """
    Evaluates the trained model.

    Args:
        processed_test_data (Input[Dataset]): Input processed test data.
        test_labels_data (Input[Dataset]): Input test labels data.
        model (Input[Model]): Input trained model.
        evaluation_metrics (Output[Artifact]): Output evaluation metrics.
    """
    import os
    import pandas as pd
    import joblib
    import json
    import tensorflow as tf
    from sklearn.metrics import classification_report

    # Load processed data and the model
    X_test = joblib.load(processed_test_data.path)
    y_test = joblib.load(test_labels_data.path)
    #_model = joblib.load(model.path)
    
    # Determine the model file path
    h5_model_path = os.path.join(model.path, "model.h5")
    # Check which format is used and load the model
    if os.path.exists(h5_model_path):
        _model = tf.keras.models.load_model(h5_model_path)
        print(f"Loaded model from: {h5_model_path}")
    else:
        raise FileNotFoundError(f"No model found in {input_model.path}")   
    
    
    NUM_CLASSES=10

    # Create a dictionary for each type of label 
    labels = {0 : "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
          5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}
    
    y_pred = _model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)
    y_true_classes = y_test.argmax(axis=1)    
    
#     from keras.layers import TFSMLayer

#     # Load the SavedModel exported with model.export()
#     _model = TFSMLayer(model.path, call_endpoint="serving_default")

#     # Wrap as a Keras model to make predictions
#     model_for_inference = tf.keras.Sequential([_model])

  
#     # Predict using wrapped TFSMLayer model
#     raw_output = model_for_inference(X_test)

#     # Extract the tensor from the dict
#     if isinstance(raw_output, dict):
#         raw_output = list(raw_output.values())[0]  # Get first output

    # y_pred_classes = tf.argmax(raw_output, axis=1).numpy()
    # y_true_classes = tf.argmax(y_test, axis=1).numpy()    

#     # Predict
#     y_pred = model_for_inference(X_test)
#     y_pred_classes = tf.argmax(y_pred, axis=1).numpy()
#     y_true_classes = tf.argmax(y_test, axis=1).numpy()  
    
    # Print classification report
    target_names = ["Class {} ({}) :".format(i,labels[i]) for i in range(NUM_CLASSES)]
    print(classification_report(y_true_classes, y_pred_classes, target_names=target_names))
    
    # Generate the classification report
    metrics_report = classification_report(
        y_true_classes, 
        y_pred_classes, 
        target_names=target_names, 
        output_dict=True
    )

    # Save metrics to the artifact
    with open(evaluation_metrics.path, "w") as f:
        json.dump(metrics_report, f)

    print(f"Evaluation metrics: {metrics_report}")    
