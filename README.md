# Fruit Quality Detector

A deep learning-based web application that classifies apples as Healthy, Rotten, or Scab-affected using a CNN model built with TensorFlow and deployed with Streamlit.

## Overview

Fruit quality inspection is a critical aspect of the agricultural and food industry, ensuring that only healthy, consumable produce reaches consumers. Traditionally, this process is manual, time-consuming, and prone to human error.

This project introduces an automated fruit quality detection system that leverages computer vision and deep learning to classify apples into three categories: Healthy, Rotten, and Scab-affected. The application uses a Convolutional Neural Network (CNN) model trained on real apple images to make accurate predictions.

Users can upload an apple image through a simple web interface, and the system instantly provides a prediction along with a health advisory and detailed explanation.

This solution aims to assist farmers, vendors, and supply chain inspectors by reducing manual effort and improving consistency in fruit quality assessment.

## Demo

Here is a quick look at the Fruit Quality Detection Web App in action:

![Fruit Quality Detection Demo](https://your-image-or-gif-link-here)

- Upload an image of an apple.
- The system classifies it as Healthy, Rotten, or Scab.
- It then provides a visual advisory and explanation based on the prediction.

> 🔗 *Live deployment (coming soon or insert Streamlit link if hosted)*


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

The **Apple Disease Dataset** used in this project is a custom dataset, created to classify apples into three categories: **Healthy**, **Rotten**, and **Scab-affected**. Due to the large size of the dataset, it has been uploaded to **Kaggle** for easy access and download.

You can explore and download the dataset directly from the following link:

```plaintext
👉 [Download the Apple Disease Dataset from Kaggle](https://www.kaggle.com/your-link)


## 📦 Installation Guide

To set up the project on your local machine, follow these steps:

### 1. Create a Python Virtual Environment

```bash
python -m venv venv
