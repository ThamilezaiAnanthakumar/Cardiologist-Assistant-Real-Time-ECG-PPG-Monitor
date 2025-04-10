# 🫀 Smart Biomedical Device for Cardiologists | ECG + PPG Monitoring System

A user-friendly, real-time cardiac monitoring system designed using Raspberry Pi 4B, biosensors, and Streamlit. This biomedical device aids cardiologists in tracking vital cardiac signals and parameters for early-stage heart disease detection and analysis.

---

## 🛠️ Project Overview

\textit{
\begin{itemize}
    \item Designed a user-friendly biomedical device targeted for cardiologists to monitor cardiac signals in real-time.
    \item Acquired ECG and PPG signals using biosensors (ECG electrodes and IR-based PPG sensor), with an ESP32/Raspberry Pi 4B for wireless transmission and processing.
    \item Developed and implemented digital filters (band-pass and notch) to eliminate baseline wander and powerline interference in real-time.
    \item Extracted key cardiac features:
    \begin{itemize}
        \item T, P, and R peak detection from ECG
        \item Heart Rate (HR) and Heart Rate Variability (HRV)
        \item Pulse Transit Time (PTT) calculated using ECG and PPG delay
        \item Blood Pressure (BP) estimation based on PTT
    \end{itemize}
    \item Developed a responsive \texttt{Streamlit} web app to visualize:
    \begin{itemize}
        \item Real-time ECG and PPG signals
        \item Live vital sign updates and auto-analysis
        \item Patient profile inputs and dynamic alert thresholds
    \end{itemize}
    \item Enabled early detection of cardiac risks like arrhythmia, abnormal BP, and reduced HRV.
    \item Hardware system developed using Raspberry Pi 4B, connected with ECG & PPG sensors for data acquisition and wireless interfacing.
    \item Future plans include emotion recognition from HRV and machine learning–based heart risk classification.
\end{itemize}
}

---

## 🖼️ System Architecture

> 📌 Add your hardware/system architecture block diagram here  
> **Example placeholder:**

![Hardware Architecture](./images/device_diagram.png)

---

## 🌐 Live Demo

Access the deployed web app below:

👉 **[Open Streamlit Web App](https://your-streamlit-app-link)**

---

## 💻 Tech Stack

- **Hardware:** Raspberry Pi 4B, ECG sensor, IR PPG sensor
- **Language:** Python
- **Signal Processing:** NumPy, SciPy, NeuroKit2
- **Frontend:** Streamlit
- **Deployment:** Localhost / Streamlit Cloud
- **Optional:** Matplotlib / Plotly for enhanced visuals

---


