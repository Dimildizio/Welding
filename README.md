# Welding Joint Size Prediction for Streamlit

![Welding](https://raw.githubusercontent.com/Dimildizio/Welding/main/production/data/welding_1.gif)


This repository contains a Streamlit application designed to predict the size of welding joints using machine learning. 

The application utilizes linear regression models to infer the depth and width of the joint based on input parameters. 

The project is structured to run both locally and within a Docker container.


## Structure

> **data/:** Contains the dataset (ebw_data.csv) and gif images (welding_1.gif, welding_3.gif) for the app.

> **models/:** Stores the trained machine learning models (linear_model_y1.plk, linear_model_y2.plk).

> **src/:** Source code for the Streamlit app and machine learning models.

> **src/app.py:** The Streamlit app interface.

> **src/main.py:** Handles the main loop and model training.

> **src/model.py:** Defines the machine learning models and their functions.

> **src/preprocessing.py:** Functions for data preprocessing.

> **Dockerfile:** Defines the Docker container setup.

> **requirements.txt:** Lists the Python dependencies for the project.


## Installation

To set up the project, follow these steps:

Clone the repository to your local machine.

Navigate to the repository directory and install the necessary dependencies using:

```
pip install -r requirements.txt
```

## Usage

Run the Streamlit application using the following command:

```
streamlit run src/app.py
```


## Docker

Alternatively, you can build and run the application using Docker:

```
docker build -t welding-production .
docker run -p 8501:8501 welding-production
```

## Contributing

Please read **CONTRIBUTING.md** for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the **LICENSE** file for details.

## Acknowledgments

Depth and width predictions are made using Linear Regression models trained on the provided dataset.

The project uses Streamlit for the web interface, making it interactive and user-friendly.
