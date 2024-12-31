import argparse
from google.cloud import aiplatform
from kfp import compiler
from pipeline import product_style_pipeline

compiler.Compiler().compile(
    pipeline_func=product_style_pipeline,
    package_path="fashion_mnist_pipeline.json"
)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Vertex AI Pipeline Job")
    parser.add_argument("--project", required=True, help="GCP Project ID")
    parser.add_argument("--location", required=True, help="GCP Location (region)")
    parser.add_argument("--data_bucket", required=True, help="GCS Bucket for data")
    parser.add_argument("--data_folder", required=True, help="Folder in the data bucket")
    parser.add_argument("--train_file", required=True, help="Training data file")
    parser.add_argument("--test_file", required=True, help="Testing data file")
    parser.add_argument("--pipeline_template", default="fashion_mnist_pipeline.json", help="Path to the pipeline template JSON")
    parser.add_argument("--pipeline_root", default="gs://fashion-mnist/fashion_mnist_classification_pipeline", help="GCS path for pipeline root")

    args = parser.parse_args()

    # Initialize Vertex AI
    aiplatform.init(
        project=args.project,
        location=args.location,
    )

    # Submit the pipeline job
    pipeline_job = aiplatform.PipelineJob(
        display_name="fashion-mnist-classification",
        template_path=args.pipeline_template,
        pipeline_root=args.pipeline_root,
        parameter_values={
            "project": args.project,
            "location": args.location,
            "data_bucket": args.data_bucket,
            "data_folder": args.data_folder,
            "train_file": args.train_file,
            "test_file": args.test_file,
        },
        enable_caching=True  # Enable caching
    )

    pipeline_job.run(sync=True)

if __name__ == "__main__":
    main()

