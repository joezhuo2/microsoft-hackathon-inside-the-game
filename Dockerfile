FROM python:3.13-slim

WORKDIR /app
COPY requirements-mcp.txt .
RUN pip install --no-cache-dir -r requirements-mcp.txt

COPY src ./src
COPY data/roster.json data/roster.json
COPY output/games_0-2 output/games_0-2
COPY output/games_3-5 output/games_3-5

RUN useradd --create-home app
USER app

ENV PYTHONPATH=/app/src HOST=0.0.0.0 PORT=8000 PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-m", "football_analytics.mcp_server"]
