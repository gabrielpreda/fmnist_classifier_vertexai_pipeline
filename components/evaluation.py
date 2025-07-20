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
