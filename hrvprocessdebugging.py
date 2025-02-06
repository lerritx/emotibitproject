import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
import os
import numpy as np

def debug_data(df, stage=""):
    """ฟังก์ชันสำหรับตรวจสอบข้อมูลในแต่ละขั้นตอน"""
    print(f"\n=== การตรวจสอบข้อมูล {stage} ===")
    print(f"จำนวนแถว: {len(df)}")
    print("ชื่อคอลัมน์:", df.columns.tolist())
    print("\nตัวอย่างข้อมูล 5 แถวแรก:")
    print(df.head())
    print("\nข้อมูลสถิติพื้นฐาน:")
    print(df.describe())
    print("=== จบการตรวจสอบ ===\n")


def process_data(file_path):
    try:
        # ตรวจสอบไฟล์
        if not os.path.exists(file_path):
            print(f"ไม่พบไฟล์ที่: {file_path}")
            return None

        print(f"กำลังอ่านไฟล์จาก: {file_path}")
        df = pd.read_csv(file_path)
        debug_data(df, "หลังจากอ่านไฟล์ CSV")

        # ตรวจสอบคอลัมน์ที่จำเป็น
        required_columns = ['LocalTimestamp', 'PG']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"ไม่พบคอลัมน์ที่จำเป็น: {missing_columns}")
            return None

        # แปลง LocalTimestamp เป็น datetime
        df['datetime'] = pd.to_datetime(df['LocalTimestamp'], unit='s')

        # ตรวจสอบและทำความสะอาดข้อมูล PG
        print("\nการตรวจสอบข้อมูล PG:")
        print("ค่า null:", df['PG'].isnull().sum())
        print("ค่าต่ำสุด:", df['PG'].min())
        print("ค่าสูงสุด:", df['PG'].max())

        # ทำความสะอาดสัญญาณ PPG
        try:
            PPG = nk.ppg_clean(df['PG'].astype(float), sampling_rate=25)
            print("\nการตรวจสอบสัญญาณ PPG หลังทำความสะอาด:")
            print("จำนวนค่า:", len(PPG))
            print("ค่าต่ำสุด:", np.min(PPG))
            print("ค่าสูงสุด:", np.max(PPG))
            df['PPG_clean'] = PPG
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการทำความสะอาดสัญญาณ PPG: {str(e)}")
            return None

        debug_data(df, "หลังจากเพิ่มคอลัมน์ PPG_clean")
        return df

    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการประมวลผลข้อมูล: {str(e)}")
        return None


def calculate_hrv_for_interval(df, start_seconds, end_seconds):
    try:
        print(f"\n=== การคำนวณ HRV สำหรับช่วง {start_seconds}-{end_seconds} วินาที ===")

        start_time = df['datetime'].min() + pd.Timedelta(seconds=start_seconds)
        end_time = df['datetime'].min() + pd.Timedelta(seconds=end_seconds)

        segment = df[(df['datetime'] >= start_time) & (df['datetime'] < end_time)].copy()
        print(f"จำนวนข้อมูลในช่วงเวลา: {len(segment)} แถว")

        if len(segment) == 0:
            print("ไม่มีข้อมูลในช่วงเวลาที่กำหนด")
            return pd.DataFrame()

        # ทำความสะอาดและ normalize สัญญาณ
        ppg_signal = segment['PPG_clean'].values
        ppg_signal = (ppg_signal - np.mean(ppg_signal)) / np.std(ppg_signal)

        print("\nสถิติของสัญญาณหลัง normalize:")
        print(f"Mean: {np.mean(ppg_signal):.6f}")
        print(f"Std: {np.std(ppg_signal):.6f}")
        print(f"Min: {np.min(ppg_signal):.6f}")
        print(f"Max: {np.max(ppg_signal):.6f}")

        # คำนวณ peaks ด้วยพารามิเตอร์ที่ปรับแต่ง
        try:
            # ใช้ clean_ppg เพื่อกรองสัญญาณก่อน
            cleaned_ppg = nk.ppg_clean(ppg_signal, sampling_rate=25, method='elgendi')
            peaks, info = nk.ppg_peaks(cleaned_ppg,
                                       sampling_rate=25,
                                       method='elgendi',  # ใช้ elgendi algorithm
                                       show=True)  # แสดงกราฟเพื่อ debug

            # Convert peaks to array if it's a DataFrame
            if isinstance(peaks, pd.DataFrame):
                peaks = peaks['PPG_Peaks'].values if 'PPG_Peaks' in peaks.columns else peaks.iloc[:, 0].values

            print(f"\nจำนวน peaks ที่พบ: {np.sum(peaks)}")

            # ตรวจสอบความถี่ของ peaks
            peak_indices = np.where(peaks)[0]
            if len(peak_indices) >= 2:
                intervals = np.diff(peak_indices) / 25  # แปลงเป็นวินาที
                heart_rate = 60 / np.mean(intervals)  # คำนวณ heart rate
                print(f"Estimated Heart Rate: {heart_rate:.1f} BPM")
                print(f"Mean RR Interval: {np.mean(intervals):.3f} seconds")

        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการหา peaks: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return pd.DataFrame()

        # คำนวณ HRV ด้วยพารามิเตอร์ที่ปรับแต่ง
        try:
            hrv_welch = nk.hrv_frequency(
                peaks,
                sampling_rate=25,
                show=True,  # แสดงกราฟ power spectrum
                psd_method="welch",
                vlf=(0.0033, 0.04),  # Very low frequency range
                lf=(0.04, 0.15),  # Low frequency range
                hf=(0.15, 0.4),  # High frequency range
                normalize=True  # Normalize values
            )

            print("\nผลลัพธ์ HRV:")
            print(hrv_welch)

            if hrv_welch.empty or hrv_welch.isnull().all().all():
                print("Warning: ไม่สามารถคำนวณค่า HRV ได้")
                return pd.DataFrame()

        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการคำนวณ HRV: {str(e)}")
            return pd.DataFrame()

        # สร้างผลลัพธ์
        hrv_results = []

        # ตรวจสอบค่าก่อนใช้
        lf_value = float(hrv_welch["HRV_LF"][0]) if not pd.isna(hrv_welch["HRV_LF"][0]) else 0.0
        hf_value = float(hrv_welch["HRV_HF"][0]) if not pd.isna(hrv_welch["HRV_HF"][0]) else 0.0

        for i in range(30):
            ratio = lf_value / hf_value if hf_value != 0 else 0.0
            hrv_results.append({
                "time": i,
                "HRV_LF": lf_value,
                "HRV_HF": hf_value,
                "LF/HF Ratio": ratio
            })

        results_df = pd.DataFrame(hrv_results)
        print("\nผลลัพธ์สุดท้าย:")
        print(results_df.describe())
        return results_df

    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการคำนวณ HRV: {str(e)}")
        print(traceback.format_exc())
        return pd.DataFrame()

def plot_hrv_interval(df_interval, title, subplot_pos):
    plt.subplot(subplot_pos)

    if not df_interval.empty:
        plt.plot(df_interval['time'], df_interval['HRV_LF'], label='HRV_LF', marker='o')
        plt.plot(df_interval['time'], df_interval['HRV_HF'], label='HRV_HF', marker='o')
        plt.plot(df_interval['time'], df_interval['LF/HF Ratio'], label='LF/HF Ratio', marker='o')

    plt.title(title)
    plt.xlabel('Time (seconds)')
    plt.ylabel('HRV Values')
    plt.grid(True)
    plt.legend()


def create_hrv_analysis(file_path):
    # ประมวลผลข้อมูล
    df = process_data(file_path)

    if df is not None:
        # คำนวณ HRV สำหรับแต่ละช่วงเวลา
        interval1 = calculate_hrv_for_interval(df, 30, 60)
        interval2 = calculate_hrv_for_interval(df, 90, 120)
        interval3 = calculate_hrv_for_interval(df, 150, 180)

        # สร้างกราฟ
        plt.figure(figsize=(15, 10))

        # กราฟแยกสำหรับแต่ละช่วงเวลา
        plot_hrv_interval(interval1, 'HRV Analysis (30-60 seconds)', 221)
        plot_hrv_interval(interval2, 'HRV Analysis (90-120 seconds)', 222)
        plot_hrv_interval(interval3, 'HRV Analysis (150-180 seconds)', 223)

        # กราฟเปรียบเทียบ
        plt.subplot(224)
        if not (interval1.empty or interval2.empty or interval3.empty):
            plt.plot(interval1['time'], interval1['LF/HF Ratio'], label='30-60s', marker='o')
            plt.plot(interval2['time'], interval2['LF/HF Ratio'], label='90-120s', marker='o')
            plt.plot(interval3['time'], interval3['LF/HF Ratio'], label='150-180s', marker='o')

        plt.title('LF/HF Ratio Comparison')
        plt.xlabel('Time (seconds)')
        plt.ylabel('LF/HF Ratio')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()
    else:
        print("ไม่สามารถสร้างกราฟได้เนื่องจากไม่มีข้อมูล")


# ใช้งานฟังก์ชัน
# ต้องระบุ path แบบเต็มหรือ relative path ที่ถูกต้อง
file_path = "~/Desktop/dataemotibit/31-01-25/neck/2025-01-31_14-46-13-914820_PG.csv"
# แปลง ~ เป็น path เต็ม
file_path = os.path.expanduser(file_path)
create_hrv_analysis(file_path)