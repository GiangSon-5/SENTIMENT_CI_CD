# 🎯 Sentiment Analysis CI/CD

---

## 🎯 1. Bài toán

**📌 Mô tả vấn đề:**
Đây là một hệ thống phân tích cảm xúc (sentiment analysis) nhị phân trên văn bản, mục tiêu chuyển văn bản thô thành nhãn cảm xúc (ví dụ Positive/Negative) để hỗ trợ đánh giá phản hồi khách hàng và giám sát ý kiến cộng đồng.

**💡 Lý do hình thành:**
Tự động hoá việc phân loại cảm xúc giúp tiết kiệm thời gian, cho phép giám sát theo thời gian thực và cải thiện quyết định kinh doanh dựa trên dữ liệu người dùng. Hệ thống tích hợp CI/CD đảm bảo pipeline training và deployment tự động hoá, giảm thiểu lỗi thủ công và tăng độ tin cậy của mô hình trong production.

---

## 📊 2. Dữ liệu

**📦 Nguồn dữ liệu:**
Dữ liệu mẫu nằm trong `data/raw/sentiment_data.csv` (dạng CSV). Hệ thống hỗ trợ:
- Tải dữ liệu từ GitHub (bộ dataset công khai Twitter sentiment nếu có sẵn)
- Fallback về dữ liệu dummy (generated sample) nếu download thất bại
- Dữ liệu nội bộ hoặc dataset công khai đã chuẩn hoá

**📐 Đặc điểm dữ liệu:**
- Định dạng: CSV (text + nhãn)
- Kích thước: ~1000 dòng (hoặc 400 dòng nếu dùng dummy data)
- Loại: tập văn bản đơn dòng (single-line text) kèm nhãn nhị phân
- Lưu trữ: file CSV (`data/raw/`) và các artifact đã xử lý dưới `data/processed/` (pickle format)

**🏷️ Giải thích tiêu đề cột:**
- `text` — Nội dung văn bản cần phân tích (ví dụ: "I love this product, absolutely amazing!")
- `target` — Nhãn cảm xúc nhị phân:
  - `1` = Positive (cảm xúc tích cực)
  - `0` = Negative (cảm xúc tiêu cực)
- `id` (nếu có) — Mã định danh mẫu dữ liệu, giúp theo dõi từng record
- `timestamp` (nếu có) — Thời điểm thu thập dữ liệu (ISO format)

**🧼 Cách xử lý dữ liệu:**

| Bước | Chi tiết |
|------|---------|
| **1. Ingest** | Tải/sinh dữ liệu CSV từ `data/raw/sentiment_data.csv` |
| **2. Làm sạch** | Chuyển thường hóa (`lower()`), xóa URL (`http\S+`), xóa ký tự đặc biệt (`[^a-zA-Z\s]`) |
| **3. Vectorization** | TF-IDF vectorizer fit trên train set, transform test set |
| **4. Tách dữ liệu** | Split 80/20 (train/test) theo `test_size: 0.2` trong config |
| **5. Lưu artifact** | Pickle format: `X_train.pkl`, `X_test.pkl`, `y_train.pkl`, `y_test.pkl` → `data/processed/` |

**⚠️ Thách thức dữ liệu:**
- 🔴 **Cân bằng nhãn**: Có khả năng mất cân bằng giữa Positive/Negative samples
- 🔴 **Dữ liệu nhiễu**: Các ký tự lạ, emoji, URL cần xóa bỏ
- 🔴 **Độ dài text không đồng nhất**: Từ 1-2 words đến hơn 50 words
- 🔴 **Đồng bộ text cleaning**: Regex pattern phải giống nhau giữa preprocess và inference
- 🔴 **Thiếu giá trị**: Một số cell có thể bị trống, cần xử lý NaN

---

## 🧠 3. Pipeline thực hiện

### **3.1 Xác định vấn đề**
- **Input**: Chuỗi văn bản tự do (raw string)
- **Output**: Nhãn cảm xúc (Positive/Negative) + độ tin cậy (confidence score 0-1)
- **Yêu cầu chức năng**:
  - Xử lý văn bản tiếng Anh
  - Dự đoán nhãn trong thời gian thực (< 100ms)
  - Cung cấp confidence score cùng dự đoán

### **3.2 Phân tích & chuẩn hóa dữ liệu**
- Đọc CSV từ `data/raw/sentiment_data.csv` (xử lý encoding)
- Áp dụng regex cleaning (lowercase, remove URLs, special chars)
- Fit `TfidfVectorizer` trên train set:
  - `max_features: 5000` (giới hạn vocabulary)
  - Tự động học IDF weights từ dữ liệu
- Transform X_train/X_test sang sparse matrix
- Lưu vectorizer dưới dạng pickle → `models/vectorizer.pkl`

### **3.3 Lựa chọn công cụ & thuật toán**
- **Mô hình**: `LogisticRegression` (sklearn)
  - Hyperparameter: `C=1.0` (inverse regularization strength)
  - Điểm mạnh: nhanh, interpretable, phù hợp với dữ liệu binary
- **Vectorizer**: `TfidfVectorizer` (sklearn)
  - TF-IDF weighting cho phép mô hình học được mức độ quan trọng của từng từ
- **Framework**: scikit-learn (nhẹ, không cần GPU)

### **3.4 Huấn luyện & kiểm thử**
1. **Load pickle data** từ `data/processed/`
2. **Khởi tạo mô hình** LogisticRegression với config từ `config/config.yaml`
3. **Fit mô hình** trên X_train
4. **Dự đoán** trên X_test
5. **Tính metrics**:
   - Accuracy = số dự đoán đúng / tổng số mẫu
   - F1-Score (weighted) = trung bình có trọng số của precision/recall
6. **Lưu mô hình** → `models/model.pkl`
7. **Log vào MLflow**:
   - Experiment: `Sentiment_Analysis_Production`
   - Run name: `Train_YYYY-MM-DD_HH-MM_<commit-hash>`
   - Artifacts: model folder + vectorizer
   - Metrics: accuracy, f1_score

### **3.5 Triển khai giải pháp**
- **API FastAPI** (`api/app.py`):
  - POST `/predict` — nhận JSON `{"text": "..."}`, trả về `{"label": "...", "confidence": ...}`
  - GET `/` — health check
- **Load mô hình** (module-level init trong `src/inference.py`):
  - Đọc `models/model.pkl` và `models/vectorizer.pkl` khi import
  - Không fit lại mô hình tại runtime
- **Docker container**:
  - Image: `ghcr.io/<org>/<repo>/sentiment-api:latest`
  - Mount volume `models/` từ project root
  - Chạy `uvicorn api.app:app --host 0.0.0.0 --port 8000`

### **3.6 Đánh giá kết quả**
- So sánh metrics (accuracy, F1) với baseline hoặc version cũ
- Kiểm tra MLflow dashboard để xem trend của các runs
- Test dự đoán trên các sample test từ miền khác nhau (domain transfer)
- Nếu metrics < ngưỡng kỳ vọng, có thể điều chỉnh hyperparameters hoặc dữ liệu

---

OK 👍 mình viết lại **phiên bản GỌN – ĐÚNG REPORT – KHÔNG DÀI DÒNG**, **chỉ hướng dẫn tạo thư mục & chạy**, **không giải thích lan man**.
Bạn **copy nguyên khối này dán vào report** là ổn.

---

# 🧩  Project Setup & Run Instructions

---

## 🧩  Khởi tạo cấu trúc dự án

```bash
# Tạo folder cha
mkdir SENTIMENT_CI_CD
cd SENTIMENT_CI_CD
```

```bash
# Tạo thư mục
mkdir -p data/raw data/processed src api models config tests .github/workflows

# Tạo file source code
touch src/__init__.py src/ingest.py src/preprocess.py src/train.py \
      src/evaluate.py src/inference.py

# Tạo file API
touch api/__init__.py api/app.py

# Tạo file config & CI/CD
touch config/config.yaml
touch .github/workflows/ci.yml

# Tạo file test
touch tests/__init__.py tests/test_preprocess.py tests/test_api.py

# Tạo file cấu hình khác
touch Dockerfile docker-compose.yaml requirements.txt
```


---

## 🛠️  Công cụ sử dụng

| Công cụ | Phiên bản | Vai trò |
|---------|---------|--------|
| **Python** | 3.9 | Runtime chính |
| **pandas** | 2.0.3 | Đọc/ghi CSV, xử lý dữ liệu |
| **scikit-learn** | 1.3.0 | TfidfVectorizer, LogisticRegression, metrics |
| **MLflow** | 2.7.1 | Tracking experiment, logging model/artifacts |
| **FastAPI** | 0.103.1 | Framework API (REST) |
| **uvicorn** | 0.23.2 | ASGI server chạy FastAPI |
| **Docker** | Latest | Build/run container (ghcr.io) |
| **pytest** | 7.4.2 | Unit test framework |
| **PyYAML** | 6.0.1 | Parse config.yaml |
| **GitHub Actions** | (CI/CD) | Orchestrate pipeline: ingest → preprocess → train → deploy → build docker |

### **Thư viện bổ trợ:**
- `httpx==0.24.1` — Test API (async HTTP client)


## ✅  Kết quả

### **📈 Tóm tắt kết quả đạt được:**
1. ✅ **Pipeline CI/CD hoàn chỉnh**:
   - Ingest → Preprocess → Train → Serve
   - Tự động trigger trên push to `main` branch
   - Self-hosted runner trên WSL (Windows Subsystem for Linux)

2. ✅ **Model training & evaluation**:
   - Model: LogisticRegression trained on vectorized text
   - Metrics logged to MLflow: accuracy + f1_score (weighted)
   - Artifacts saved: `models/model.pkl`, `models/vectorizer.pkl`

3. ✅ **Production deployment**:
   - FastAPI endpoint: `POST /predict`
   - Docker image pushed to GHCR (GitHub Container Registry)
   - Volume mount cho models/ directory
   - Ready for serving inference requests

4. ✅ **MLflow tracking**:
   - Experiment: `Sentiment_Analysis_Production`
   - Run history stored in `mlruns/` (persistent across CI runs)
   - Commit hash + timestamp in run names for traceability

### **📝 Kết quả chi tiết (cập nhật sau mỗi training run):**
Xem chi tiết metrics trong `mlruns/` hoặc qua MLflow UI

---


## 🧩 Chạy hệ thống bằng Docker (Production)

### (Tùy chọn) Login GitHub Container Registry (GHCR)

```bash
echo ${{ secrets.GITHUB_TOKEN }} | docker login ghcr.io \
  -u GiangSon-5 --password-stdin
```

*(Bỏ qua bước này nếu repository ở chế độ public)*

### Chạy container API

```bash
docker run -d \
  -p 8000:8000 \
  --name sentiment-api-prod \
  -v /mnt/c/Users/Admin/Desktop/sentiment_ci_cd/models:/app/models \
  ghcr.io/giangson-5/sentiment_ci_cd/sentiment-api:latest
```

---

## 🧩 Kiểm tra API inference

Sau khi container chạy thành công, gửi request dự đoán sentiment bằng `curl`:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I really love this product"}'
```

API trả về kết quả gồm:

* Nhãn cảm xúc (Positive / Negative)
* Confidence score của dự đoán





