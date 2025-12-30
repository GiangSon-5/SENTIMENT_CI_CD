# Giai đoạn 1: Build Image để chạy (Runtime)
FROM python:3.9-slim

# Thiết lập biến môi trường để log mượt hơn
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Cài đặt thư viện
# Chỉ copy requirements.txt trước để tận dụng Docker Cache
COPY requirements.txt .
# Trong thực tế, file requirements này nên là file rút gọn (chỉ chứa fastapi, uvicorn, scikit-learn, numpy...)
# không cần chứa mlflow, pytest, pandas (nếu lúc chạy không dùng pandas).
# Nhưng để đơn giản ta dùng chung file.
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy Code
# Lưu ý: Lúc này thư mục models/ ở máy build (GitHub Actions) đã có file .pkl
# do bước Train trước đó tạo ra.
COPY ./src ./src
COPY ./api ./api
COPY ./config ./config
COPY ./models ./models 

# Expose port
EXPOSE 8000

# Tạo user non-root để bảo mật (Best Practice)
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Chạy API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]