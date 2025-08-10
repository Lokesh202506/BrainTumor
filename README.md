# Brain Tumor Detection Using X-Ray Images

A Python GUI application for detecting brain tumors from X-ray images using deep learning (CNN). The app allows dataset upload, preprocessing, model training, tumor segmentation, and classification, all via a user-friendly interface.

## Features
- Upload and preprocess X-ray image datasets
- Train and evaluate a CNN model
- Segment and highlight tumor regions
- Classify new images as "Tumor Detected" or "No Tumor Detected"
- Visualize training accuracy and loss
- GUI built with Tkinter

## Project Structure
- `BrainTumor.py`: Main application script
- `brain_tumor_dataset/`: Sample images (yes/no tumor)
- `Model/`: Pre-trained models, weights, and training history
- `testImages/`: Images for testing

## Requirements
- Python 3.x
- Keras
- TensorFlow
- OpenCV
- scikit-learn
- Tkinter
- matplotlib
- numpy
- pickle

## Getting Started
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install keras tensorflow opencv-python scikit-learn matplotlib numpy
   ```
3. Run the application:
   ```bash
   python BrainTumor.py
   ```
4. Use the GUI to upload datasets, preprocess, train, and classify images.

## Screenshots
Add screenshots of the GUI and results here.

## License
This project is open source and available under the MIT License.
