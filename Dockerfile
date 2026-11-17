FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

# SQLite 数据落盘目录（Compose 中挂载命名卷，重启后状态保留）
ENV DB_PATH=/data/unlearn.db \
    RETRAIN_SLICE_DELAY=0.8
RUN mkdir -p /data

EXPOSE 8000

# 单 worker：重训状态机与后台线程驻留进程内
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
