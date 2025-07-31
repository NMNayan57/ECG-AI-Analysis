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
Performance: ~76.7% test accuracy (improves with quality data).
Edge Optimization: <30MB model size, <100ms inference time.

Limitations

Image analysis requires ecg_model_deployment.pth to be available.
Poor-quality ECG images may reduce accuracy.
Real-time monitoring is currently simulated.

Contributing
Contributions are welcome! Please:


Contact
Developed by Nasim Mahmud Nayan for PhD research in Healthcare AI. For inquiries, open an issue on GitHub or contact [smnoyan670@gmail.com].
