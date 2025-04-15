👁️ OpenCV Python Facial Detection & Recognition (Web – Flask)
This project is a **lightweight**, **real-time face detection and recognition system** built with **Python**, **OpenCV**, and **Flask**. It uses webcam input and compares detected faces with saved facial data for instant recognition — **no deep learning models or heavy training involved**.

Currently, the project runs in a **web browser**, with planned support for **Android** in future releases.

✨ Features

 • 🎯 Real-time face detection using OpenCV

 • 👤 Lightweight facial recognition using saved face data (no deep learning)

 • 💡 Fast and efficient: no heavy model training

 • 📸 Add new faces dynamically to the dataset

 • 🖥️ Simple, clean Flask-based web interface

 • 📱 Android support planned (via mobile client or REST API)

⚙️ Technologies Used

 • Python 3

 • Flask

 • OpenCV (cv2)

 • NumPy

 • sklearn (KNN method)

 • HTML5 

🚀 Getting Started

🔧 Prerequisites

-Python 3.x

-pip

-A webcam

-virtualenv (optional)

📦 Installation

1) Clone the repo

 ``` bash
git clone https://github.com/your-username/opencv-face-recognition-flask.git
cd opencv-face-recognition-flask
```

2) (Optional) Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3) Install dependencies

```bash
pip install flask flask-cors scikit-learn pillow opencv-python-headless numpy
```


🧠 How It Works

📸 Face detection is performed using OpenCV Haar cascades.

🧬 When a new face is registered, the facial encoding and name are extracted and saved in a .pkl file (using Python’s pickle module).

📂 This .pkl file stores all known users' face data (encodings) and their corresponding names — acting as the facial database.

🧠 A KNN classifier (from scikit-learn) is optionally used to recognize users based on this stored data.

🔍 During runtime, incoming face encodings from webcam input are compared with the ones in the .pkl file for instant recognition.

✅ No deep learning or cloud processing involved — the system is lightweight, fully local, and fast.

📁 Project Structure

```csharp
opencv-face-recognition-flask/
├── static/             # CSS/JS/Images
├── templates/          # HTML files
├── data/               # Saved facial images Web
│   └── haarcascade_frontalface_default.xml   # Haar cascade classifier
├── android/            # Saved facial data (Android - future support)
└── test.py              # Flask main app
```

📽️ Video Demo

<a data-start="223" data-end="337" rel="noopener" target="_new" class="" href=""><img alt="Watch the Demo" data-start="224" data-end="289" src="https://img-c.udemycdn.com/course/480x270/2756342_cfca_13.jpg" style="max-width:100%;width:110vh;"></a>

📱 Android Support (Coming Soon)
We're planning to add Android integration via a mobile client or using API endpoints to allow mobile face detection and recognition using the same system.
