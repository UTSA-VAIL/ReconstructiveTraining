# Reconstructive Training for Real-World Robustness in Image Classification
This is a Tensorflow 2 implementation for our DNOW Workshop paper "Reconstructive Training for Real-World Robustness in Image Classification".

## Docker Environment
Reconstructive training uses a Docker container to train the defense. We have provided a Docker container at https://hub.docker.com/repository/docker/utsavisionailab/reconstructivetraining.

We have also provided the Dockerfile if you prefer to build the image manually.
```sh
cd ReconstructiveTraining
docker build -t <youruser>/reconstructivetraining . 
```

Once done, run the Docker image. Below is an example command to run the Docker container.
```sh
docker run --gpus all --rm --shm-size 64G -it -u $(id -u):$(id -g) -v "$(pwd)":/app utsavisionailab/reconstructivetraining:latest
```

## Training
To train the defense, first we need to warm-up the generator using the following command:
```sh
python3 generator.py \
--original_input_dir PATH_TO_ORIGINAL_IMAGES_DIRECTORY \
--attacked_input_dir PATH_TO_ATTACKED_IMAGES_DIRECTORY \
--weights_output_dir PATH_TO_STORE_MODEL_WEIGHTS \
--logs_output_dir PATH_TO_STORE_TENSORBOARD_LOGS \
--image_size HEIGHT WIDTH COLOR \
--batch_size BATCH_SIZE \
--epochs NUM_OF_EPOCHS_TO_TRAIN \
```

Before we can finish training, we need to set-up a discriminator config file. Here would be an example config file for VGG19.
```json
{
    "module": "vgg19",
    "model": "VGG19",
    "weights": "imagenet",
    "classes": 1000,
    "image_width": 224,
    "image_height": 224,
    "image_channels": 3,
    "clip_min": [
        -103.939,
        -116.779,
        -123.68
    ],
    "clip_max": [
        151.061,
        138.22101,
        131.32
    ],
    "mode": "caffe",
    "verbose": 1,
    "workers": 36
}
```


After we warm-up the generator and made a discriminator config file, we are ready to attach the target model and fully train the defense.
```sh
python3 gan.py \
--original_input_dir PATH_TO_ORIGINAL_IMAGES_DIRECTORY \
--attacked_input_dir PATH_TO_ATTACKED_IMAGES_DIRECTORY \
--generator_input_dir PATH_WHERE_GENERATOR_MODEL_WEIGHTS_WERE_STORED \
--weights_output_dir PATH_TO_STORE_MODEL_WEIGHTS \
--logs_output_dir PATH_TO_STORE_TENSORBOARD_LOGS \
--discriminator_config_file PATH_TO_DISCRIMINATOR_CONFIG \
--labels_file PATH_TO_LABELS_FILE \
--num_classes NUM_OF_UNIQUE_CLASSES \
--image_size HEIGHT WIDTH COLOR \
--batch_size BATCH_SIZE \
--epochs NUM_OF_EPOCHS_TO_TRAIN
```

## Evaluation
In order to evaluate the defense, you need to run the following command:
```sh
python eval_defense.py \
--input_dir PATH_TO_IMAGES_DIRECTORY \
--labels_file PATH_TO_LABELS_FILE \
--discriminator_config_file PATH_TO_DISCRIMINATOR_CONFIG \
--weights_dir PATH_WHERE_MODEL_WEIGHTS_WERE_STORED \
--image_size HEIGHT WIDTH COLOR \
--num_classes NUM_OF_UNIQUE_CLASSES
```

## Citations
If you are using our code in a publication, please use the citation provided below
    
    @inproceedings{icesurface2018wacv,
        title = {Reconstructive Training for Real-World Robustness in Image Classification},
        author = {David Patrick and Michael Geyer and Richard Tran and Amanda Fernandez},
        booktitle = {???},
        year = {2022}
    }  
