import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import scipy.optimize as opt
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Cardiac Health Monitoring System",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
    <style>
        body {
            background-image: url('https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.pinterest.com%2Fpin%2Ffree-stock-photo-of-human-heart-anatomical-rendering-on-dark-background--699535754615618586%2F&psig=AOvVaw1DAq5FVQDWOcCUPnFRa00a&ust=1742838942829000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCJjU-5PjoIwDFQAAAAAdAAAAABAE'); /* Add your own background image */
            background-size: cover;
            color: white;
        }
        .sidebar .sidebar-content {
            background-color: #1f1f1f;
            color: white;
        }
        h1 {
            font-family: 'Roboto', sans-serif;
            color: #ff6f61;
        }
        h2, h3 {
            color: #ff6f61;
        }
        .stButton>button {
            background-color: #ff6f61;
            color: white;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #ff3b30;
        }
        .stTextInput>div>input {
            background-color: #333;
            color: white;
        }
        .stFileUploader>label {
            color: white;
        }
    </style>
""", unsafe_allow_html=True)



with st.sidebar:
    selected = option_menu(
        "Menu", ["Home", "About", "Contact"],
        icons=["house", "info-circle", "envelope"],
        menu_icon="cast", default_index=0
    )
st.title("📊 Cardiac Health Monitoring System")
st.image("heart.jpg", use_container_width=True)
ecg_rate = st.number_input("Enter ECG Sampling Rate", min_value=1, value=250)
ppg_rate = st.number_input("Enter PPG Sampling Rate", min_value=1, value=250)
        
def upload_and_process_ecg():
    #st.image("heart.jpg", use_container_width=True)
    ecg_file = st.file_uploader("Upload ECG CSV File", type="csv")
    ppg_file = st.file_uploader("Upload PPG CSV File", type="csv")
    
    if ecg_file is not None and ppg_file is not None:
        ecg_data = pd.read_csv(ecg_file).to_numpy()
        ppg_data = pd.read_csv(ppg_file).to_numpy()
        
        #ecg_rate = st.number_input("Enter ECG Sampling Rate", min_value=1, value=250)
        #ppg_rate = st.number_input("Enter PPG Sampling Rate", min_value=1, value=250)
        
        time = np.linspace(0, len(ecg_data)/ecg_rate, len(ecg_data))
        
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time, ecg_data, label="ECG Signal")
        ax.set_title("ECG Signal Visualization")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid()
        st.pyplot(fig)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time, ppg_data, label="PPG Signal", color='red')
        ax.set_title("PPG Signal Visualization")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid()
        st.pyplot(fig)
        
        return ecg_data, ppg_data  #, ecg_rate, ppg_rate
    return None, None, None, None

def process_ecg(ecg_data, ecg_rate):
    #st.write(ecg_rate)
    if ecg_data.ndim > 1:
        ecg_data = ecg_data[:, 0]
    signals, info = nk.ecg_process(ecg_data, sampling_rate=ecg_rate)
    r_peaks = info["ECG_R_Peaks"]
    st.write(r_peaks)
    delineate_signal, delineate_info = nk.ecg_delineate(ecg_data, rpeaks=r_peaks, sampling_rate=ecg_rate, method="dwt")
    
    p_onsets = delineate_info["ECG_P_Onsets"]
    r_onsets = delineate_info["ECG_R_Onsets"]
    pr_intervals = np.array(r_onsets) - np.array(p_onsets)
    pr_intervals = pr_intervals[~np.isnan(pr_intervals)]
    pr_interval = np.mean(pr_intervals / ecg_rate) if len(pr_intervals) > 0 else 0
    
    return pr_interval, r_peaks

def classify_av_block(pr_interval, r_peaks, fs=ecg_rate):
    st.write(ecg_rate)
    rr_intervals = np.diff(r_peaks) / ecg_rate
    
    if pr_interval > 0.2 and all(rr_intervals > 0.6):  
        return "First-Degree AV Block"
    for i in range(1, len(rr_intervals)):
        if rr_intervals[i] > 1.2:
            return "Second-Degree AV Block (Mobitz I)"
    if pr_interval > 0.2 and any(rr_intervals > 1.5):
        return "Second-Degree AV Block (Mobitz II)"
    if np.std(rr_intervals) > 0.5:  
        return "Third-Degree AV Block"
    return "Normal ECG Pattern"

def classify_heart_rate(heart_rate):
    if heart_rate > 100:
        return f"Tachycardia: {heart_rate}"
    if heart_rate < 60:
        return f"Bradycardia: {heart_rate}"
    return f"Normal: {heart_rate}"

def calibrate():
    # Calibration for PTT, SBP, and DBP using uploaded files
    ptt_file_c = st.file_uploader("Upload PTT Calibration CSV File", type="csv")
    sbp_file_c = st.file_uploader("Upload SBP Calibration CSV File", type="csv")
    dbp_file_c = st.file_uploader("Upload DBP Calibration CSV File", type="csv")
    
    if ptt_file_c is not None and sbp_file_c is not None and dbp_file_c is not None:
        ptt_values_c = pd.read_csv(ptt_file_c).to_numpy()
        sbp_values_c = pd.read_csv(sbp_file_c).to_numpy()
        dbp_values_c = pd.read_csv(dbp_file_c).to_numpy()
        ptt_values_c = ptt_values_c.flatten()
        sbp_values_c = sbp_values_c.flatten()
        dbp_values_c = dbp_values_c.flatten()
        st.write(f"Is array_1d 1D? {dbp_values_c.ndim == 1}")
        st.write(ptt_values_c)
        st.write(f"Is array_1d 1D? {ptt_values_c.ndim == 1}")
        st.write(f"Is array_1d 1D? {sbp_values_c.ndim == 1}")

        if np.any(np.isnan(ptt_values_c)) or np.any(np.isinf(ptt_values_c)):
            st.write("ptt_values_c contains NaN or inf values")
        else:
            st.write("ok")
        st.write(type(ptt_values_c)) 

        if np.any(np.vectorize(lambda x: not isinstance(x, (int, float)))(ptt_values_c)):
            st.write("ptt_values_c contains non-numeric values")
        else:
            st.write("ptt_values_c contains only numeric values")
        st.write(ptt_values_c.shape)  # Should print (n,) where n > 1
    

        
        def sbp_model(ptt, a, b, c):
            return a * (ptt ** -b) + c

        def dbp_model(ptt, d, e, f):
            return d * (ptt ** -e) + f

        try:
           #params_sbp, _ = opt.curve_fit(sbp_model, ptt_values_c, sbp_values_c, p0=[1, 1, 100], maxfev=10000)
            params_sbp, _ = opt.curve_fit(sbp_model, ptt_values_c, sbp_values_c, p0=[1, 1, 100], maxfev=10000, bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
            st.write("Fitting successful:", params_sbp)
        except Exception as e:
            st.write("Error during curve fitting:", e)


        # Fit the models to find a, b, c (for SBP) and d, e, f (for DBP)
        params_sbp, _ = opt.curve_fit(sbp_model, ptt_values_c, sbp_values_c, p0=[1, 1, 100], maxfev=10000, bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
        params_dbp, _ = opt.curve_fit(dbp_model, ptt_values_c, dbp_values_c, p0=[1, 1, 60], maxfev=10000, bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))

        a, b, c = params_sbp
        d, e, f = params_dbp

        st.write(f"Calibration successful! SBP model: SBP = {a:.2f} * PTT^(-{b:.2f}) + {c:.2f}")
        st.write(f"Calibration successful! DBP model: DBP = {d:.2f} * PTT^(-{e:.2f}) + {f:.2f}")
        
        return a, b, c, d, e, f
    else:
        st.warning("Please upload the calibration files to proceed.")
        return None, None, None, None, None, None

def main():
    ecg_data, ppg_data  = upload_and_process_ecg() #, ecg_rate, ppg_rate
    
    if ecg_data is not None:
        pr_interval, r_peaks = process_ecg(ecg_data, ecg_rate)
        classification = classify_av_block(pr_interval, r_peaks, ecg_rate)
        heart_rate = 60 / np.mean(np.diff(r_peaks) / ecg_rate)
        heart_rate_classification = classify_heart_rate(heart_rate)
        
        st.subheader("ECG Analysis Results")
        st.write(f"AV Block Classification: {classification}")
        st.write(f"Heart Rate: {heart_rate:.2f} BPM ({heart_rate_classification})")
        
        if pr_interval == 0:
            st.warning("Calibration Required: Please check ECG signal quality.")
        
        a, b, c, d, e, f = calibrate()  # Calibrate for PTT, SBP, DBP
        
        if a is not None and b is not None and c is not None:
            ptt_value = np.mean(np.diff(ppg_data))  # Example for PTT calculation (replace with actual calculation)
            predicted_sbp = a * (ptt_value ** -b) + c
            predicted_dbp = d * (ptt_value ** -e) + f
            st.write(f"Predicted Blood Pressure: {predicted_sbp:.2f}/{predicted_dbp:.2f} mmHg")
        
        st.sidebar.header("Contact")
        st.sidebar.write("For support, contact: Tamil | Email: tamil@example.com")


if selected == "Home":
    #st.title("📊 Cardiac Health Monitoring System")
    if __name__ == "__main__":
        main()
# About Section
elif selected == "About":
    st.title("📌 About")
    st.write(
        "This skin disease prediction app is designed to assist healthcare professionals and individuals "
        "in identifying common skin diseases. The model is trained on a diverse dataset and can predict "
        "the following conditions with high accuracy:"
    )
    st.write("- Cellulitis\n- Impetigo\n- Athlete's Foot\n- Nail Fungus\n- Ringworm")
    st.write("- Cutaneous Larva Migrans\n- Chickenpox\n- Shingles\n- Normal Skin")
    st.write(
        "This project leverages convolutional neural networks (CNNs) for accurate classification "
        "and aims to provide quick preliminary analysis, aiding in faster medical intervention."
    )

# Contact Section
elif selected == "Contact":
    st.title("📞 Contact")
    st.write(
        "For inquiries, collaborations, or feedback, please reach out to us through the following channels:" 
    )
    st.write("- **Email:** thamilezaiananthakumar@gmail.com")
    st.write("- **Phone:** +940762934089")
    st.write("- **GitHub:** [Thamilezai Ananthakumar](https://github.com/ThamilezaiAnanthakumar)")
    st.write("- **LinkedIn:** [Thamilezai Ananthakumar](https://www.linkedin.com/in/thamilezai-ananthakumar-387a922a4)")

# Footer
st.markdown("""
---
Developed with ❤️ by Thamilezai Ananthakumar
""")
