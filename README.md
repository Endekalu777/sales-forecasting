# Sales Prediction Project

This project focuses on predicting sales using machine learning and deep learning techniques, specifically employing a Random Forest model and an LSTM (Long Short-Term Memory) neural network. The analysis involves feature engineering, data cleaning, model training, and forecasting future sales based on historical data.

## Project Overview

The core functionality of the project is encapsulated within the `SalesPrediction` class. This class is designed to handle the entire sales prediction workflow, including data preprocessing, model training, and forecasting sales. The main components include:

- **Feature Engineering**: Creating additional features from the date and other categorical variables to enhance model performance.
- **Data Preprocessing**: Cleaning and scaling the data, ensuring it is in a suitable format for modeling.
- **Model Training**: Training both Random Forest and LSTM models for sales prediction.
- **Forecasting**: Generating future sales predictions based on the trained models.
- **Model Serialization**: Saving trained models for future use.

## Installation

### Creating a Virtual Environment

#### Using Conda

If you prefer Conda as your package manager:

1. Open your terminal or command prompt.

2. Navigate to your project directory.

3. Run the following command to create a new Conda environment:

    ```bash
    conda create --name your_env_name python=3.12.5
    ```
    - Replace `your_env_name` with the desired name for your environment (e.g., `sales_forecasting`) and `3.12.5` with your preferred Python version.

4. Activate the environment:

    ```bash
    conda activate your_env_name
    ```

#### Using Virtualenv

If you prefer using `venv`, Python's built-in virtual environment module:

1. Open your terminal or command prompt.

2. Navigate to your project directory.

3. Run the following command to create a new virtual environment:

    ```bash
    python -m venv your_env_name
    ```
    - Replace `your_env_name` with the desired name for your environment.

4. Activate the environment:

    - On Windows:
        ```bash
        .\your_env_name\Scripts\activate
        ```

    - On macOS/Linux:
        ```bash
        source your_env_name/bin/activate
        ```

### Installing Dependencies with pip

Once your virtual environment is created and activated, you can install packages and run your Python scripts within this isolated environment. Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Installing Dependencies with Conda

Alternatively, you can use Conda to install the project dependencies. Note that you will need to install each package individually. To do this, first ensure that you have activated your Conda environment, then use the following commands to install each required package:

```bash
conda install -c conda-forge package-name
```

### Clone this package
- To install the network_analysis package, follow these steps:

- Clone the repository:

```bash
git clone https://github.com/your-username/sales-forecasting.git
```
- Navigate to the project directory:

```bash
cd Solar_radiation_analysis
Install the required dependencies:
```

```bash
pip install -r requirements.txt
```

## Usage Instructions

Once the dependencies are installed, you can run the analysis notebooks by launching Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

## Running the Flask App
- Ensure You Have a templates Folder: Create a folder named templates within your project directory, and place the home.html file within it.
- Run the Flask App: Navigate to your project directory in your terminal and run:
- python run app.py (application name)
- This will start the Flask development server. The app should be accessible in your web browser at http://127.0.0.1:5000/.

## Contributions
Contributions are welcome! If you find any issues or have suggestions for improvement, feel free to create a pull request or open an issue.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

### Contact
For any questions or additional information please contact Endekalu.simon.haile@gmail.com
