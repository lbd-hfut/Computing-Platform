# Computing-Platform: PINN-DIC

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)

A brief description of your project, its purpose, and what it aims to achieve.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

### Prerequisites

- Python 3.7 or higher
- Required packages are listed in `requirements.txt`.

### Clone the Repository

```bash
git clone https://github.com/lbd-hfut/Computing-Platform.git
cd Computing-Platform
```

## Usage

To use this project, follow the instructions below.

### Prepare the Dataset to be solved

- Please place the reference image, deformed image, and ROI image into any subfolder within the `data` directory. The ROI image can be created by running the `utils/select_roi.py` script, which allows users to manually select the ROI in either a circular or rectangular shape based on their needs. For more complex ROI shapes, you can use the built-in Windows `mspaint` software. In MSPaint, paint the ROI area white and cover any obvious white spots in the background with black.

![mspaint image](mspaint.png)

- Please name the reference image, deformed image, and ROI image in the following format:
  * Reference image: The file name starts with the letter `"r"` followed by a number (e.g. `r0000.bmp`).
  * Deformed image: The file name starts with the letter `"d"` followed by a number (e.g. `d0001.bmp, d0002.bmp`).
  * ROI image: The file name starts with `"mask"` followed by a number (e.g. `mask0003.bmp`).

The numbers in the file name should be in order, and the file extension can be .bmp, .JPG, or .png.


### Running the Application

First, select the deep learning parameters in the `configs/config.py` file, then you can run the main application script with the following command:

```bash
python train.py
python plot_fig.py
```

## Features

- **Physics-Informed Neural Networks (PINN) Integration**: Our method, PINN-DIC, leverages Physics-Informed Neural Networks to solve the Digital Image Correlation (DIC) problem, combining the strengths of deep learning with the principles of physics.
  
- **No Manual Parameter Setting**: Unlike traditional Subset-DIC, our approach does not require manual parameter tuning, making the process more efficient and user-friendly.

- **Point-by-Point Full-Field Solution**: The PINN-DIC method solves the displacement field for the entire image domain point-by-point, providing a comprehensive analysis of the deformation field.

- **High Accuracy in Non-Uniform Field Measurements**: Our method achieves higher accuracy, particularly in scenarios involving non-uniform deformation fields, making it suitable for complex experimental setups.

- **Precise Handling of Irregular Boundaries**: The PINN-DIC method excels in solving images with irregular boundaries, offering high precision in boundary deformation measurements.

- **No Need for Training Datasets**: Unlike supervised learning DIC methods, PINN-DIC does not require pre-existing datasets for training, allowing for immediate application to a wide range of problems.

- **Lightweight Neural Network Architecture**: The method uses a simple fully connected neural network, which is more lightweight than those used in unsupervised learning DIC, leading to faster computations and higher accuracy.

- **Ease of Integration**: The PINN-DIC method is designed to be easily integrated with other numerical inversion techniques, enhancing its versatility and applicability in various fields.

## Project Structure

```bash
Computing-Platform/
├── configs/             # Configuration files for the project
├── data/                # Data and scripts for data processing
├── docs/                # Documentation and related resources
├── demos/               # Demo scripts showcasing example usage
├── layers/              # Custom layers and loss functions
├── logs/                # Logging output and training logs
├── utils/               # Utility scripts and helper functions
├── weights/             # Model weights and checkpoints
├── LICENSE              # License file for the project
├── README.md            # Project overview and usage instructions
├── requirements.txt     # List of Python dependencies
├── train.py             # Script to train the model and solve displacement
└── plot_fig.py          # Script to plot and save displacement figures
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

Boda Li, School of Ocean and Civil Engineering, Shanghai Jiao Tong University, Shanghai 200240, China.

Email: `leebda_sjtu@sjtu.edu.cn`
