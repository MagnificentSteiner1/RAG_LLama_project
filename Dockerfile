FROM python:3.11.14
#komentar za probu, EC2 instance running , EC2HOSt = IP adress
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY RAG_env.txt .
RUN pip install --no-cache-dir -r RAG_env.txt

COPY API/ ./API
COPY Agent/ ./Agent

COPY scripts/ ./scripts
COPY startup.sh ./startup.sh
RUN chmod +x /app/startup.sh

EXPOSE 8000

CMD ["bash", "startup.sh"]