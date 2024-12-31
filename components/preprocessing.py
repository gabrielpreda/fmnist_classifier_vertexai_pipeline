from kfp.v2.dsl import component, Output, Dataset


@component(base_image="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/tf2-cpu.2-17.py310")
def preprocess_data(data_bucket: str,
                    data_folder: str,
                    train_file: str,
                    test_file: str,
                    processed_train_data: Output[Dataset],
                    processed_test_data: Output[Dataset],
                    train_labels_data: Output[Dataset],
                    test_labels_data: Output[Dataset]
                   ):
    """
    Preprocess the data.

    Args:
        data_bucket (str): Bucket where the data is stored.
        data_folder (str): Folder where the data is stored.
        train_file (str): Train data file.
        test_file (str): Test data file.
        processed_train_data (Output[Dataset]): Path to the train data.
        processed_test_data (Output[Dataset]): Path to the test data.
        train_labels_data (Output[Dataset]): Path to the train labels.
        test_labels_data (Output[Dataset]): Path to the test labels.
    """
        
    # import packages
    from google.cloud import storage
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from keras.utils import to_categorical
    import joblib
    import io
    
    NUM_CLASSES=10
    IMG_ROWS=28
    IMG_COLS=28
    
    client = storage.Client()
    bucket = client.bucket(data_bucket)
    
    def read_data(bucket, data_folder, data_file):
        """
        Function to read data from a GCP bucket
        Args:
            bucket: bucket where the data is stored
            data_folder: folder where the data is stored
            data_file: data file (either train or test data)
            
        Returns:
            _df: dataset with images pixels and labels
        """
        
        blob_path = f"{data_folder}/{data_file}"
        blob = bucket.blob(blob_path)
        csv_content = blob.download_as_text()
        _df = pd.read_csv(io.StringIO(csv_content))
    
        return _df
    
    def data_preprocessing(raw):
        """
        Function to preprocess the (train/test) data
        Separates label data and pixels data
        Format pixels data in the 28 x 28 pixels size
        Args:
            raw: raw data (either train or test)
        Returns:
            out_x: array with shaped images data
            out_y: array with labels
        """
        out_y = to_categorical(raw.label, NUM_CLASSES)
        num_images = raw.shape[0]
        x_as_array = raw.values[:,1:]
        x_shaped_array = x_as_array.reshape(num_images, IMG_ROWS, IMG_COLS, 1)
        out_x = x_shaped_array / 255
        return out_x, out_y

    # Read train and test data
    train_data_df = read_data(bucket, data_folder, train_file)
    test_data_df = read_data(bucket, data_folder, test_file)

    # Preprocess the data
    X_train, y_train = data_preprocessing(train_data_df)
    X_test, y_test = data_preprocessing(test_data_df)
    
    # Write files    
    joblib.dump(X_train, processed_train_data.path)
    joblib.dump(X_test, processed_test_data.path)
    joblib.dump(y_train, train_labels_data.path)
    joblib.dump(y_test, test_labels_data.path)

    

