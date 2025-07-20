from kfp.v2.dsl import component, Input, Model, Artifact

@component(base_image="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/tf2-cpu.2-17.py310")
def register_model(
    project: str,
    location: str,
    model: Input[Model], 
    metrics: Input[Artifact],
):
    """
    Registers a model in Vertex AI Model Registry with a group name based on style.
    
    Args:
        project (str): Project ID.
        location (str): GCP location (e.g. `us-central1`).
        model (Input[Model]): Input trained model artifact.
        metrics (Input[Artifact]): Input metrics report.
        
    """
    import os
    import json
    import joblib
    import time
    from pathlib import Path
    import tensorflow as tf
    from google.cloud import aiplatform, storage
    
    # Initialize aiplatform
    aiplatform.init(project=project, location=location)
    
    # Load evaluation metrics from the artifact
    with open(metrics.path, "r") as f:
        evaluation_metrics = json.load(f)
            
    
    # Add labels; evaluation metrics are added to description
    labels = {
        "model_type": "classifier"
    }
    print(labels)
    
    # Register the model
    model_display_name = f"fmnist-classifier"
    #First load model
    h5_model_path = os.path.join(model.path, "model.h5")

    # Check which format is used and load the model
    if os.path.exists(h5_model_path):
        _model = tf.keras.models.load_model(h5_model_path)
        print(f"Loaded model from: {h5_model_path}")
    else:
        raise FileNotFoundError(f"No model found in {model.path}")

    # Export model to GCS in saved_model.pb format
    _model.export(f"{model.path}/output_model")

    print(f"Exported saved_model.h5  model to: {model.uri}/output_model")

    time.sleep(20)
    # Detect the parent model
    # List all models with the same display name
    models = aiplatform.Model.list(filter=f'display_name="{model_display_name}"')
    
    # Get the resource name of the parent model
    parent_model = None
    if models:
        parent_model = models[0].resource_name  # Assuming the first match is your target model
        print(f"Parent Model Resource Name: {parent_model}")
    else:
        print("No existing model found with the specified display name.")
        
    # Register the model
    uploaded_model = aiplatform.Model.upload(
        artifact_uri=f"{model.uri}/output_model",  # Model artifact path
        display_name=model_display_name,  # Display name for the model
        parent_model=parent_model,  # Use style as group name
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-13:latest", # Container image
        labels=labels # Attach model type metrics as labels
    )

    print("Model uploaded succesfully.")
    
    # Add description to the model
    beautified_json = json.dumps(evaluation_metrics, indent=4)
    print(beautified_json)
    
    uploaded_model.update(
        description=f"FMNIST Image Classifier: Evaluation Metrics:\n{beautified_json}"
    )

    print(f"Model registered with display name '{model_display_name}'")
    print(f"Model registered with ID: {uploaded_model.resource_name}")
    print(f"Evaluation metrics: {evaluation_metrics}")