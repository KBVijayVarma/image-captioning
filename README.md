# Image Caption Generator

## Overview

This project is an Image Caption Generator that utilizes a Pretrained ResNet-50 model for feature extraction from images and an LSTM (Long Short-Term Memory) model to generate captions for those images. The model has been trained on the COCO 2017 dataset, which contains a diverse collection of images and corresponding captions. The entire pipeline is deployed in a [Streamlit](https://captionify.streamlit.app/) app, allowing users to upload an image and receive a generated caption.

## Table of Contents

1. [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Clone the Repository](#clone-the-repository)
    - [Installing Dependencies](#installing-dependencies)
2. [Training](#training)
    - [Dataset Preparation](#dataset-preparation)
    - [Training the Model](#training-the-model)
3. [Running Inference](#running-inference)
    - [Launching the Streamlit App](#launching-the-streamlit-app)
    - [Image Input](#image-input)
4. [Project Structure](#project-structure)

## Getting Started

### Prerequisites

Create a virtual environment in Python using the `venv` module.

1. Open a terminal or command prompt.

2. Navigate to the directory where you want to create the virtual environment. You can use the `cd` command to change your directory. For example:

    ```
    cd path/to/your/desired/directory
    ```

3. Once you are in the desired directory, run the following command to create a virtual environment:

    On macOS and Linux:

    ```
    python3 -m venv venv_name
    ```

    On Windows (using Command Prompt):

    ```
    python -m venv venv_name
    ```

    Replace `venv_name` with the name you want to give to your virtual environment. For example:

    ```
    python3 -m venv myenv
    ```

4. Activate the virtual environment:

    On macOS and Linux:

    ```
    source venv_name/bin/activate
    ```

    On Windows (using Command Prompt):

    ```
    venv_name\Scripts\activate
    ```

    After activation, your command prompt or terminal will show the virtual environment name, indicating that you are now working within the virtual environment.

### Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/KBVijayVarma/image-captioning.git
cd image-captioning
```

### Installing Dependencies

Install the required packages using pip in the Virtual Environment:

```bash
pip install -r requirements.txt
```

## Training

### Dataset Preparation

Before training the model, you need to prepare the COCO 2017 dataset.

Download the following from the [COCO](https://cocodataset.org/#download) Website.

1. [2017 Train images [118K/18GB]](http://images.cocodataset.org/zips/train2017.zip)

2. [2017 Val images [5K/1GB]](http://images.cocodataset.org/zips/val2017.zip)

3. [2017 Test images [41K/6GB]](http://images.cocodataset.org/zips/test2017.zip)

4. [2017 Train/Val annotations [241MB]](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

5. [2017 Testing Image info [1MB]](http://images.cocodataset.org/annotations/image_info_test2017.zip)

Unzip the above files into a folder coco_dataset. Refer the [Project Structure](#project-structure)

### Training the Model

-   Create a folder `models` in the working directory

-   Run the [training.ipynb](training.ipynb) for training the Image Captioning Model

-   Rename the final pickle (.pkl) files in the `models` folder to `encoder.pkl` and `decoder.pkl`

## Running Inference

### Launching the Streamlit App

To use the Image Caption Generator, launch the Streamlit app:

```bash
streamlit run app.py
```

### Image Input

In the Streamlit app, input the Image using the following options:

-   URL of the Image
-   File Uploader
-   Camera

## Project Structure

```
image-captioning/
│
├── imgcaptioning/
│   ├── coco_dataset.py
│   ├── data_loader.py
│   ├── model.py
│   ├── inference_pipeline.py
│   ├── tokenizer.py
│   ├── utils.py
│   └── vocabulary.py
│
├── models/
│   ├── encoder.pkl
│   └── decoder.pkl
│
├── coco_dataset/
│   ├── annotations/
│   │   ├── captions_train2017.json
│   │   ├── captions_val2017.json
│   │   ├── image_info_test-dev2017.json
│   │   ├── image_info_test2017.json
│   │   ├── instances_train2017.json
│   │   ├── instances_val2017.json
│   │   ├── person_keypoints_train2017.json
│   │   └── person_keypoints_val2017.json
│   │
│   ├── train2017/
│   ├── test2017/
│   └── val2017/
│
├── .gitignore
├── app.py
├── requirements.txt
├── training.ipynb
└── vocab.json
```
