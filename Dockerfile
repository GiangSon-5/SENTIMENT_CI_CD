# Sử dụng Python Slim image cho nhẹ
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements trước để tận dụng Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code vào
COPY . .

# Set biến môi trường để Python không tạo file .pyc
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose port cho API
EXPOSE 8000

# Chạy script ingest, preprocess, train trước khi start API 
# (Trong thực tế nên tách stage build và run, nhưng ở đây gộp để demo đơn giản: container khởi động là có model mới nhất)
# Tuy nhiên, chuẩn nhất là Train ở CI/CD, còn Docker chỉ chứa Model đã train. 
# Ở bài này: Chúng ta sẽ copy model đã train từ bên ngoài vào hoặc train khi build.
# Để đơn giản cho CI/CD: Chúng ta sẽ Build image CHỨA SẴN model.

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]