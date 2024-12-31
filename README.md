# VertexAI Kubeflow Pipeline for Fashion MNIST classifier training

The VertexAI Kubeflow pipeline to train and test a model for classifying Fashion MNIST data.

## Prerequisites

Create your own project on GCP. 
Activate all APIs for VertexAI.
Create a bucket. In this project, the bucket name is "fashion-mnist". 
Access and copy to your own data bucket the Fashion MNIST data from Kaggle: https://www.kaggle.com/datasets/zalando-research/fashionmnist
In this project, the two csv data files with train and test data from the above mentioned dataset are copied in the dedicated bucket in a "data" folder.

Create then a VertexAI Workbench instance (the lowest spec available will be fine, since you are just starting the pipeline from Workbench).

## Code

The code structure is as following:

'''
components  
    |  
    preprocessing.py  
    training.py  
    evaluation.py  
    register.py  
pipeline.py  
run_pipeline.py  
'''

The components (`preprocessing.py`, `training.py`, `evaluation.py`, `register.py`) are stored in a `components` folder. The pipeline code is in `pipeline.py`. And the script to initialize and start the pipeline is in `run_pipeline.py`).

## Run pipeline

To run the pipeline, execute the following code from a bash console:
```
python run_pipeline.py \
    --project=<YOUR_PROJECT_ID> \
     --location=<YOUR_LOCATION> \
     --data_bucket="fashion-mnist" \
     --data_folder="data" \
     --train_file="fashion-mnist_train.csv" \
     --test_file="fashion-mnist_test.csv"
```

You can navigate then to VertexAI/pipelines and monitor the pipeline run.
