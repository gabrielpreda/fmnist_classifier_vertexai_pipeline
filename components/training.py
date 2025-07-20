from kfp.v2.dsl import component, Input, Output, Dataset, Model


@component(base_image="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/tf2-cpu.2-17.py310")
def train_model(processed_train_data: Input[Dataset],
                train_labels_data: Input[Dataset],
                model: Output[Model]):

    """
    Train the model.

    Args:
        processed_train_data (Input[Dataset]): Input processed train data.
        train_labels_data (Input[Dataset]): Input train labels data.
        model (Output[Model])): Output trained model.
    """
    import os
    import pandas as pd
    import joblib
    import tensorflow as tf
    from tensorflow.keras import layers, models, Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
    from tensorflow.keras.losses import categorical_crossentropy
    NUM_CLASSES=10
    IMG_ROWS=28
    IMG_COLS=28

    # Load the processed data from the path
    X_train = joblib.load(processed_train_data.path)
    y_train = joblib.load(train_labels_data.path)

    def define_model():

        # Model
        _model = Sequential()
        
        _model.add(Conv2D(filters=32, 
                          kernel_size=(3, 3),
                          activation='relu',
                          kernel_initializer='he_normal',
                          input_shape=(IMG_ROWS, IMG_COLS, 1)))
        _model.add(MaxPooling2D((2, 2)))
        _model.add(Dropout(0.25))
        
        _model.add(Conv2D(filters=64, 
                          kernel_size=(3, 3), 
                          activation='relu'))
        _model.add(MaxPooling2D(pool_size=(2, 2)))
        _model.add(Dropout(0.25))
        
        _model.add(Conv2D(filters=128, 
                          kernel_size=(3, 3), 
                          activation='relu'))
        _model.add(Dropout(0.4))
        
        _model.add(Flatten())
        _model.add(Dense(128, activation='relu'))
        _model.add(Dropout(0.3))
        
        _model.add(Dense(NUM_CLASSES, activation='softmax'))


        _model.compile(loss=categorical_crossentropy,
                      optimizer='adam',
                      metrics=['accuracy'])

        return _model

    def train_model(_model, X_train, y_train, batch_size=32, epochs=50, validation_split=0.2):

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        history = _model.fit(
            X_train, 
            y_train, 
            batch_size=batch_size, 
            epochs=epochs, 
            validation_split=validation_split,
            callbacks=[early_stopping]
        )

        return _model, history
    
    # define and compile the model
    _model = define_model()
    
    # run the model
    _model, history = train_model(_model, X_train, y_train, epochs=20)

    # Save the trained model
    #joblib.dump(_model, model.path)
    save_model_path = os.path.join(model.path, "model.h5")
    _model.save(save_model_path, save_format="h5")
    print(f"Model saved to: {save_model_path}")
