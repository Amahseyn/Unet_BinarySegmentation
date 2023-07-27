# Segmentation using SMP (Segmentation Model Pytorch)

This repository contains the code implementation for image segmentation using SMP (Semantic Multi-Pooling) technique with PyTorch. SMP is a powerful and state-of-the-art method for semantic segmentation in computer vision tasks. The code provided in this repository is designed to train and evaluate the SMP models on a custom dataset.
## Installation
Before running the code, you need to install the required dependencies. Please follow these steps:

1.Install PyTorch according to the instructions provided in this [link](https://pytorch.org/get-started).

2.Install *segmentation-models-pytorch* with the following command:

    pip install segmentation-models-pytorch

3.Install *numpy* with the following command:

    pip : pip install segmentation-models-pytorch
    conda :  step1:"conda config --env --add channels conda-forge" step2:"conda install numpy"

4.Install *pillow* with the following command:

    pip : step1:"python3 -m pip install --upgrade pip" step2:"python3 -m pip install --upgrade Pillow"
    conda : "conda install -c anaconda pillow"
## Dataset Preparation
Ensure that your dataset directory has the following structure:


    path_to_your_directory/

        train/
            0/
                image1.png
                image2.png
            1/
                image1.png
                image2.png

        train_mask/
            0/
                image1.png
                image2.png
            1/
                image1.png
                image2.png
        validation/
            0/
                image1.png
                image2.png
            1/
                image1.png
                image2.png

        validation_mask/
            0/
                image1.png
                image2.png
            1/
                image1.png
                image2.png
        test/
            0/
                image1.png
                image2.png
            1/
                image1.png
                image2.png

        test_mask/
            0/
                image1.png
                image2.png
            1/
                image1.png
                image2.png
## Usage
Follow these steps to use the SMP model for image segmentation:

1.Set the required parameters such as data_dir, image_size, batch, and epochs in the provided code.

2.Import the necessary libraries and define the SMP model using the desired encoder (e.g., resnet34).

**You can choose one of them for models :(Unet ,  UnetPlusPlus , MAnet, Linknet , FPN, PSPNet, PAN, DeepLabV3, DeepLabV3+)**
**Follow this [link](https://smp.readthedocs.io/en/latest/encoders.html) to use encoders for your models.**  

3.Create custom datasets for training, validation, and testing by using the CustomDataset class, which prepares the data for training and evaluation.

4.Define the data transformation pipelines for training and testing datasets using T.Compose.

5.Train the SMP model using the custom dataset with the chosen optimizer and loss function (Jaccard Loss for binary segmentation).

6.Evaluate the trained model on the validation dataset to monitor performance metrics such as loss and IoU.

7.Finally, test the model on the test dataset and obtain the evaluation results.
## Contributing
If you find any issues or have suggestions for improvements, please feel free to contribute by opening an issue or submitting a pull request. You can contact me via email at [mhhashemi1379@gmail.com] for any inquiries or discussions related to the project.
