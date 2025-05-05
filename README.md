# Fruit Quality Detector

A deep learning-based web application that classifies apples as Healthy, Rotten, or Scab-affected using a CNN model built with TensorFlow and deployed with Streamlit.


## Overview

Fruit quality inspection is a critical aspect of the agricultural and food industry, ensuring that only healthy, consumable produce reaches consumers. Traditionally, this process is manual, time-consuming, and prone to human error.

This project introduces an automated fruit quality detection system that leverages computer vision and deep learning to classify apples into three categories: Healthy, Rotten, and Scab-affected. The application uses a Convolutional Neural Network (CNN) model trained on real apple images to make accurate predictions.

Users can upload an apple image through a simple web interface, and the system instantly provides a prediction along with a health advisory and detailed explanation.

This solution aims to assist farmers, vendors, and supply chain inspectors by reducing manual effort and improving consistency in fruit quality assessment.


## Demo

Here is a quick look at the Fruit Quality Detection Web App in action:

![Fruit Quality Detection Demo](demo.pdf)

- Upload an image of an apple.
- The system classifies it as Healthy, Rotten, or Scab.
- It then provides a visual advisory and explanation based on the prediction.

> 🔗 *[Live Demo – Try the App](https://food-quality-detector-lhyodzh53worpc3a4ehbwh.streamlit.app/)*


## Features

- 🍎 Classifies apples into **Healthy**, **Rotten**, or **Scab**
- 🧠 Provides visual feedback and safety advisory for users
- 📷 Accepts user-uploaded apple images (JPEG/PNG)
- ⚙️ Built using **TensorFlow** for model inference
- 🌐 Web interface powered by **Streamlit**
- ✅ Clean, interactive, and informative UI


## Tech Stack

- **Python** – Core programming language
- **TensorFlow & Keras** – Deep learning model development
- **OpenCV** – Image processing and handling
- **Streamlit** – Web app development and UI
- **Jupyter Notebook** – Model training and experimentation


## 🗂️ Dataset

The **Apple Disease Dataset** used in this project is a custom dataset, created to classify apples into three categories: **Healthy**, **Rotten**. Due to the large size of the dataset, it has been uploaded to **Google Drive** for easy access and download.

You can explore and download the dataset directly from the following link:

👉 [Download the Apple Disease Dataset from Kaggle](https://drive.google.com/drive/folders/1cqV_C51BThqkjZ932wIgMwLTGcVfDer2?usp=sharing)


## 📦 Installation Guide

To set up the project on your local machine, follow these steps:

### 1. Create a Python Virtual Environment

```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

-On Windows:
  ```bash
  .\venv\Scripts\activate
  ```
-On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

### 3. Clone the Repository

```bash
git clone https://github.com/jacob-daniel-clg/food-quality-detector.git
```
```bash
cd food-quality-detector
```

### 5. Install the Required Dependencies

```bash
pip install -r requirements.txt
```


## How to Run the App

To run the web application, use the following command:

```bash
streamlit run app.py
```


## Model Info

The deep learning model used in this project is based on **MobileNet V2**, a lightweight convolutional neural network, which leverages **transfer learning** for efficient performance. The model is stored in the `applemodel.h5` file and was trained to classify apples into three categories: **Healthy**, **Rotten**.

You can find the full training process and model development in the linked training notebook:

👉 [Training Notebook - fruitmodel.ipynb](https://colab.research.google.com/drive/1_Q566KLMNgmCetmkuNy9gqvQ7Osea_Jt?usp=sharing)

The model achieved an accuracy of **99%** on the test dataset, demonstrating its ability to classify apple images into the appropriate categories.

You can load the model from the `applemodel.h5` file and use it for predictions in the web app.


## Author & Acknowledgments

**Author**: Jacob Daniel R 
This project was conducted as part of the **Naan Mudhalvan** initiative. 

You can connect with me through my professional platforms:  
- [LinkedIn](www.linkedin.com/in/jacobdanielr)  
- Email: [jacobdanielr82@gmail.com]

Feel free to reach out for any inquiries or collaboration opportunities!


## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

