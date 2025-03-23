import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import scipy.optimize as opt

def upload_and_process_ecg():
    st.set_page_config(page_title="ECG & PPG Analysis", page_icon="❤️")
    st.title("📊 ECG & PPG Analysis")
    st.sidebar.header("About")
    st.sidebar.write("This application analyzes ECG and PPG signals, detecting heart rate conditions and AV blocks.")
    
    ecg_file = st.file_uploader("Upload ECG CSV File", type="csv")
    ppg_file = st.file_uploader("Upload PPG CSV File", type="csv")
    
    if ecg_file is not None and ppg_file is not None:
        ecg_data = pd.read_csv(ecg_file).to_numpy()
        ppg_data = pd.read_csv(ppg_file).to_numpy()
        
        ecg_rate = st.number_input("Enter ECG Sampling Rate", min_value=1, value=250)
        ppg_rate = st.number_input("Enter PPG Sampling Rate", min_value=1, value=250)
        
        time = np.linspace(0, len(ecg_data)/ecg_rate, len(ecg_data))
        
        st.image("heart_image.png", use_column_width=True)
        
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
        
        return ecg_data, ppg_data, ecg_rate, ppg_rate
    return None, None, None, None

def process_ecg(ecg_data, ecg_rate):
    signals, info = nk.ecg_process(ecg_data, sampling_rate=ecg_rate)
    r_peaks = info["ECG_R_Peaks"]
    delineate_signal, delineate_info = nk.ecg_delineate(ecg_data, rpeaks=r_peaks, sampling_rate=ecg_rate, method="dwt")
    
    p_onsets = delineate_info["ECG_P_Onsets"]
    r_onsets = delineate_info["ECG_R_Onsets"]
    pr_intervals = np.array(r_onsets) - np.array(p_onsets)
    pr_intervals = pr_intervals[~np.isnan(pr_intervals)]
    pr_interval = np.mean(pr_intervals / ecg_rate) if len(pr_intervals) > 0 else 0
    
    return pr_interval, r_peaks

def classify_av_block(pr_interval, r_peaks, fs=250):
    rr_intervals = np.diff(r_peaks) / fs
    
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
    ptt_file = st.file_uploader("Upload PTT Calibration CSV File", type="csv")
    sbp_file = st.file_uploader("Upload SBP Calibration CSV File", type="csv")
    dbp_file = st.file_uploader("Upload DBP Calibration CSV File", type="csv")
    
    if ptt_file is not None and sbp_file is not None and dbp_file is not None:
        ptt_values_c = pd.read_csv(ptt_file).to_numpy()
        sbp_values_c = pd.read_csv(sbp_file).to_numpy()
        dbp_values_c = pd.read_csv(dbp_file).to_numpy()
        
        def sbp_model(ptt, a, b, c):
            return a * (ptt ** -b) + c

        def dbp_model(ptt, d, e, f):
            return d * (ptt ** -e) + f

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
    ecg_data, ppg_data, ecg_rate, ppg_rate = upload_and_process_ecg()
    
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

if __name__ == "__main__":
    main()
