

FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Cài thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy Code (Chỉ copy src, api, config)
COPY ./src ./src
COPY ./api ./api
COPY ./config ./config

# --- [QUAN TRỌNG] ---
# KHÔNG COPY models ở đây nữa (vì mình muốn tách data ra ngoài)
# Thay vào đó, tạo một thư mục RỖNG tên là models
RUN mkdir -p /app/models

# 3. User & Permission
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]