ECG-AI-Analysis
Overview
The Explainable Medical Cyber-Physical System (MCPS) is an AI-powered(Deep Learning, ML , XAI) tool for analyzing ECG data through multiple modalities: real-time signal monitoring, file-based analysis, and paper ECG image analysis. Built for global health equity, it provides interpretable, lead-specific ECG analysis with a lightweight model suitable for edge deployment in low-resource settings.
Key Features

Multi-Modal Analysis: Supports real-time ECG signals, file uploads (CSV, WFDB), and paper ECG image analysis.
Explainable AI: Provides simple, detailed, or technical explanations tailored to user expertise.
Lead-Specific Insights: Identifies abnormalities in individual ECG leads with attention-based analysis.
Edge-Ready: <30MB model size, <100ms inference time, and offline capability.
Multilingual: Supports English and Bengali.
Professional Reporting: Generates downloadable ECG analysis reports.

Use Cases

Rural healthcare centers with paper ECGs
Emergency departments
Telemedicine and remote consultations
Community health screenings

Installation
Prerequisites

Python 3.8+
Git
Streamlit Community Cloud account
GitHub account

Steps

Clone the Repository:
git clone https://github.com/NMNayan57/ECG-AI-Analysis.git
cd ECG-AI-Analysis


Create a Virtual Environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Run Locally:
streamlit run enhanced_app.py



Deployment on Streamlit Community Cloud

Push to GitHub:

Initialize a Git repository:
git init
git add .
git commit -m "Initial commit"


Create a repository on GitHub named ECG-AI-Analysis.

Link and push to GitHub:
git remote add origin https://github.com/your-username/ECG-AI-Analysis.git
git branch -M main
git push -u origin main




Deploy on Streamlit Community Cloud:

Log in to Streamlit Community Cloud.
Click "New app" and connect your GitHub account.
Select the ECG-AI-Analysis repository and specify enhanced_app.py as the main file.
Click Deploy. Streamlit will handle dependency installation and host the app.


Note: Ensure ecg_model_deployment.pth is included in the repository or uploaded to Streamlit's file system for image analysis functionality.


Requirements
The project dependencies are listed in requirements.txt:
streamlit==1.29.0
numpy==1.24.3
pandas==2.0.3
plotly==5.17.0
torch==2.0.1
torchvision==0.15.2
Pillow==10.0.0

Usage

Access the App: Open the deployed app URL or run locally (streamlit run enhanced_app.py).
Select Analysis Mode:
Real-time Monitor: Simulate real-time ECG monitoring.
File Analysis: Upload CSV, DAT, or TXT files in PTB-XL or WFDB format.
Image Analysis: Upload a photo or scan of a paper ECG for lead-specific analysis.


Adjust Settings: Choose explanation level (Simple, Detailed, Technical) and language (English, Bengali) in the sidebar.
Generate Reports: Download professional ECG reports from the Image Analysis tab.

Model Details

Architecture: EfficientNet-B0 with attention mechanism and uncertainty/quality heads.
Training Data: PTB-XL dataset + generated ECG images.
Performance: ~72.7% test accuracy (improves with quality data).
Edge Optimization: <30MB model size, <100ms inference time.

Limitations

Image analysis requires ecg_model_deployment.pth to be available.
Poor-quality ECG images may reduce accuracy.
Real-time monitoring is currently simulated.

Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a Pull Request.

License
This project is licensed under the MIT License.
Contact
Developed by Nasim Mahmud Nayan for PhD research in Healthcare AI. For inquiries, open an issue on GitHub or contact [smnoyan670@gmail.com].