# CypherCam Master Violence Detection System

A computer vision desktop application built to automatically monitor and detect acts of violence in pre-recorded videos. The application features a native desktop interface that allows users to upload media files, process individual video frames through a deep learning model, and track sequence metrics to identify aggressive behavior.

## Features
- Desktop Interface: Built as a standalone desktop application with a clean, dedicated window layout and custom background styling.
- Native File Browser: Integrated system dialog window enabling quick selection of AVI, MP4, or MOV video files.
- Real-time Classification Overlay: Automatically processes video inputs frame-by-frame and renders live text classifications on top of the media player.
- Intelligent Alert Thresholding: Uses a dynamic sequential frame counter to track consecutive alerts, preventing false triggers and detecting sustained acts of violence.

## Model Processing and Execution
The pipeline processes individual video elements through specific computational steps:
1. Frame Capture and Resizing: Extracts sequential frames via open video streams and standardizes image dimension shapes to 224x224 pixels.
2. Pixel Array Normalization: Formats and scales raw image pixel data by dividing array elements by 255.0 to optimize deep network inference.
3. Neural Network Prediction: Feeds the normalized multi-dimensional arrays into a pre-trained H5 model file to extract classification tags.
4. Consecutive Tracking: Evaluates output arrays and increments a detection counter to track continuous violent actions across frames.

## Built With
- Python: The core programming language used to structure the desktop application and internal operational logic.
- Tkinter: The native graphical user interface library used to construct windows, header text fields, and functional control buttons.
- TensorFlow and Keras: The core machine learning engine utilized to load the deep neural network architecture and evaluate frame structures.
- OpenCV: The computer vision library utilized to handle media stream files, manipulate pixel arrays, and render video preview playback windows.
- Pillow: The imaging processing library used to open, scale, and display background image files natively inside the layout.
