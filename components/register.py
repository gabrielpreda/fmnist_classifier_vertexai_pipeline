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
    import json
    import joblib
    from pathlib import Path
    from google.cloud import aiplatform, storage
    
    def convert_to_string(data):
        """
        Recursively convert all fields in a dictionary to strings.
        Args:
            data (dict): The dictionary to convert.
        Returns:
            dict: A dictionary with all fields as strings.
        """
        if isinstance(data, dict):
            return {key: convert_to_string(value) for key, value in data.items()}
        elif isinstance(data, (float, int)):
            return f"{data:.4f}"
        else:
            return str(data)
    
    
    aiplatform.init(project="vector-search-quick-start", location="us-central1")
    
    # Load evaluation metrics from the artifact
    with open(metrics.path, "r") as f:
        evaluation_metrics = json.load(f)
            
    # Convert the evaluation metrics to labels
    evaluation_metrics_report_as_string = convert_to_string(evaluation_metrics)
    
    # Register the model
    model_display_name = f"fmnist-classifier"
    uploaded_model = aiplatform.Model.upload(
        artifact_uri=model.path,  # Model artifact path
        display_name=model_display_name,  # Display name for the model
        parent_model=model_display_name,  # Use style as group name
        serving_container_image_uri="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/tf2-cpu.2-17.py310", # Container image
        labels=labels # Attach evaluation metrics as labels
    )

