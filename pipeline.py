from kfp.v2.dsl import pipeline
from components.preprocessing import preprocess_data
from components.training import train_model
from components.evaluation import evaluate_model
from components.register import register_model

@pipeline(name="fashion-mnist-classification", pipeline_root="gs://fashion-mnist/fashion_mnist_classification_pipeline")
def product_style_pipeline(
    project: str,
    location: str,
    data_bucket: str, 
    data_folder: str, 
    train_file: str, 
    test_file: str):
    """
    Pipeline for train and evaluate the fashion mnist classifier model.

    Args:
        project: Input GCP project name.
        location: Input location (e.g. `us-central1`)
        data_bucket: Input bucket name.
        data_folder: Input folder path name.
        train_file: Input the train file name.
        test_file: Input the test file name.
    """
    
    # 1. Preprocess data
    preprocess_task = preprocess_data(
        data_bucket=data_bucket,
        data_folder=data_folder,
        train_file=train_file,
        test_file=test_file    
    )

    # 2. Train model
    train_task = train_model(
        processed_train_data=preprocess_task.outputs['processed_train_data'],
        train_labels_data=preprocess_task.outputs['train_labels_data']
    )
    
    # 3. Evaluate model
    evaluation_task = evaluate_model(
        processed_test_data=preprocess_task.outputs['processed_test_data'],
        test_labels_data=preprocess_task.outputs['test_labels_data'],
        model=train_task.outputs['model'],
    )
    
    # 4. Register model
    register_model(
        project=project,
        location=location,
        model=train_task.outputs['model'],
        metrics=evaluation_task.outputs['evaluation_metrics'],
    )
