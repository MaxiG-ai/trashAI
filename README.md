# trashAI
### Felix Schaefer, 5638170; Maximilian Graf, 9114848
Repository for Course AI in WiSe 2022 at DHBW Mannheim. 

## Overview
Ziel war es, mit einm RaspberryPi aufgenommene Fotos lokal zu klassifizieren. Dafür wurde mit Tensorflow ein Image Classifier trainiert und anschließend in ein Tensorflow lite Modell umgewandelt. Diese wurde auf dem RaspberryPi ausgeführt.

## Vorgehen
- Ein passender Trainingsdatensatz wurde online gesucht ([Quelle](https://github.com/garythung/trashnet))
- Training eines eigenen CNNs, dabei wurde berücksichtigt und getestet: (`EigenesModel.ipynb`)
  - verschiedene Modell-Architekturen (Layer Arten und Aufbau)
  - Drop-outs
  - Image Augmentation
- Transfer Learning auf Grundlage des ResNet50 (`TransferLearning.ipynb`)
- Konvertierung in Tensorflow lite Modell
- Hardware Setup: RaspberryPi mit Kamera und GPIO Button einrichten
- Script zur Aufnahme von Bilder mit RapsberryPi
- Tensorflow lite Model für Vorhersage verwendet

## Model Perfomance
Eigenes Model:      ca. 65% val accuracy
Transfer Learning:  ca. 85% val accuracy

## Hardware
- Raspberry Pi
- Camera Module
- lots of braincells

## Resources
- https://projects.raspberrypi.org/en/projects/getting-started-with-picamera/4
- https://projects.raspberrypi.org/en/projects/push-button-stop-motion/4
- https://techoverflow.net/2019/07/24/how-to-capture-raspi-camera-image-using-opencv-python/
- https://stackoverflow.com/questions/9427553/how-to-download-a-file-from-server-using-ssh
- https://www.tensorflow.org/overview
- https://www.tensorflow.org/lite

Added Notes:
- https://towardsdatascience.com/how-to-build-an-image-classifier-for-waste-sorting-6d11d3c9c478
- https://tamalhansda.medium.com/classify-trash-with-ai-d1dabbba55d5
- https://github.com/garythung/trashnet (Data)
