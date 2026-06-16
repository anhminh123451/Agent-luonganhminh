import pandas as pd

# 1. Đọc file CSV gốc
file_goc = "data/raw/BankFAQs.csv"
df = pd.read_csv(file_goc)

# 2. Cấu hình số hàng cho mỗi file con
so_hang_moi_file = 100

# 3. Tiến hành chia nhỏ bằng vòng lặp
# range(0, tổng số hàng, 100) sẽ chạy từ 0, 100, 200, ..., 1000
for i in range(0, len(df), so_hang_moi_file):
    # Cắt (slice) dữ liệu từ hàng i đến i + 100
    df_chunk = df.iloc[i : i + so_hang_moi_file]
    
    # Tạo tên file mới (ví dụ: output_0.csv, output_1.csv, ...)
    ten_file_con = f"data_csv/output_{i // so_hang_moi_file}.csv"
    
    # Xuất ra file CSV mới (index=False để không ghi thêm cột số thứ tự)
    df_chunk.to_csv(ten_file_con, index=False)
    print(f"Đã tạo: {ten_file_con} với {len(df_chunk)} hàng.")