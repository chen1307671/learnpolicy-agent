FROM python:3.13-slim

# 云平台会注入 PORT；默认 7860 便于 Hugging Face Spaces 直接识别
ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=7860

WORKDIR /app

# 只装服务端必需依赖（numpy / scikit-learn），避免 512MB 免费实例构建超时
COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY run.py ./
COPY src ./src
COPY raw ./raw

EXPOSE 7860

# 先监听端口、后台训练模型；/api/health 在预热期返回 loading
CMD ["python", "run.py", "--serve"]
