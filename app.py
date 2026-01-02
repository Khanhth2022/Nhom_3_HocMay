import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- 1. CẤU HÌNH DANH SÁCH MÔ HÌNH ---
MODEL_FILES = {
    "Stacking (Mô hình tổng hợp)": "stacking_model.pkl",
    "Decision Tree (Cây quyết định)": "decision_tree_model.pkl",
    "Logistic Regression (Hồi quy)": "logistic_regression_model.pkl",
    "Perceptron (Mạng nơ-ron đơn giản)": "perceptron_model.pkl"
}

st.set_page_config(page_title="Dự đoán Rời bỏ", layout="wide")

# --- 2. GIAO DIỆN CHỌN MÔ HÌNH ---
st.sidebar.header("⚙️ Cấu hình hệ thống")
selected_model_name = st.sidebar.selectbox(
    "Chọn thuật toán dự đoán:",
    list(MODEL_FILES.keys())
)

selected_file = MODEL_FILES[selected_model_name]

# --- 3. HÀM TẢI MÔ HÌNH ---
@st.cache_resource
def load_specific_model(filename):
    if not os.path.exists(filename):
        return None
    return joblib.load(filename)

# Tải dữ liệu
data = load_specific_model(selected_file)

if data is None:
    st.error(f"❌ Không tìm thấy file '{selected_file}'. Bạn đã chạy code lưu model (bước joblib.dump) chưa?")
    st.stop()

# Giải nén gói tin
current_model = data['model']
scaler = data['scaler']
model_features = data['features']
# Lấy ngưỡng (nếu file cũ không có thì mặc định 0.5)
threshold = data.get('threshold', 0.5)

st.sidebar.success(f"Đã tải: {selected_model_name}")
st.sidebar.info(f"Ngưỡng cắt (Threshold): {threshold}")

# --- 4. GIAO DIỆN NHẬP LIỆU ---
st.title("🛍️ Dự Đoán Rời Bỏ Khách Hàng")
st.markdown(f"Đang sử dụng mô hình: **{selected_model_name}**")
st.divider()

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Thời gian gắn bó (Tháng)", min_value=0, value=12)
    # --- ĐÃ SỬA: CHUYỂN THÀNH NHẬP KHOẢNG CÁCH ---
    warehouse_dist = st.number_input("Khoảng cách từ kho đến nhà (Km)", min_value=0, value=15)
    
    order_cat = st.selectbox("Danh mục hay mua", ['Laptop & Accessory', 'Mobile Phone', 'Fashion', 'Grocery', 'Others'])
    complain = st.selectbox("Có từng khiếu nại không?", [0, 1], format_func=lambda x: "Có" if x == 1 else "Không")

with col2:
    day_since_last = st.number_input("Số ngày từ lần đặt cuối", min_value=0, value=5)
    cashback = st.number_input("Tiền hoàn (Cashback)", min_value=0.0, value=150.0)
    gender = st.selectbox("Giới tính", ['Male', 'Female'])
    marital = st.selectbox("Tình trạng hôn nhân", ['Single', 'Married', 'Divorced'])

# Mở rộng (Optional)
with st.expander("Nhập thêm thông tin chi tiết (Không bắt buộc)"):
    satisfaction = st.slider("Điểm hài lòng (1-5)", 1, 5, 3)
    num_device = st.slider("Số thiết bị đăng ký", 1, 6, 2)
    pref_payment = st.selectbox("Thanh toán ưa thích", ['Debit Card', 'Credit Card', 'E wallet', 'UPI', 'COD'])
    pref_login = st.selectbox("Thiết bị hay dùng", ['Mobile Phone', 'Computer', 'Phone'])
    city_tier = st.selectbox("Cấp độ thành phố", [1, 2, 3])

# --- 5. XỬ LÝ VÀ DỰ ĐOÁN ---
if st.button("🚀 PHÂN TÍCH NGAY", type="primary"):
    # Tạo DataFrame
    input_df = pd.DataFrame({
        'Tenure': [tenure],
        'CityTier': [city_tier],
        'WarehouseToHome': [warehouse_dist], # <--- ĐÃ SỬA: Dùng biến warehouse_dist
        'PreferredPaymentMode': [pref_payment],
        'Gender': [gender],
        'NumberOfDeviceRegistered': [num_device],
        'PreferedOrderCat': [order_cat],
        'SatisfactionScore': [satisfaction],
        'MaritalStatus': [marital],
        'NumberOfAddress': [2], 
        'Complain': [complain],
        'DaySinceLastOrder': [day_since_last],
        'CashbackAmount': [cashback],
        'PreferredLoginDevice': [pref_login]
    })
    
    # One-Hot Encoding & Reindex
    input_df_encoded = pd.get_dummies(input_df)
    input_df_encoded = input_df_encoded.reindex(columns=model_features, fill_value=0)
    
    # Chuẩn hóa
    input_data_scaled = scaler.transform(input_df_encoded)
    
    # Dự đoán (Sử dụng Threshold)
    try:
        # Lấy xác suất
        prob = current_model.predict_proba(input_data_scaled)[0][1]
        prob_msg = f"(Xác suất: {prob:.1%})"
        
        # So sánh với Ngưỡng (Threshold)
        is_churn = 1 if prob >= threshold else 0
        
    except:
        # Trường hợp Perceptron không có predict_proba
        is_churn = current_model.predict(input_data_scaled)[0]
        prob_msg = "(Mô hình này không tính xác suất)"

    # Hiển thị kết quả
    st.divider()
    if is_churn == 1:
        st.error(f"⚠️ CẢNH BÁO: Khách hàng có nguy cơ RỜI BỎ! {prob_msg}")
        if threshold != 0.5:
             st.caption(f"*Kết quả dựa trên ngưỡng cắt tối ưu: {threshold}*")
    else:
        st.success(f"✅ AN TOÀN: Khách hàng sẽ TIẾP TỤC sử dụng. {prob_msg}")