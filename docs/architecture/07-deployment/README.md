# 07 - Deployment View

## Infrastructure Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Local Development Machine                    │
│                  AMD Ryzen 9 7950X / 128GB DDR5                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Docker Compose                        │   │
│  │                                                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │   TG4 App    │  │ TimescaleDB  │  │    Redis     │  │   │
│  │  │   (Python)   │  │  (Postgres)  │  │   (Cache)    │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │   │
│  │                                                          │   │
│  │  ┌──────────────┐  ┌──────────────┐                     │   │
│  │  │  Prometheus  │  │   Grafana    │                     │   │
│  │  │  (Metrics)   │  │ (Dashboard)  │                     │   │
│  │  └──────────────┘  └──────────────┘                     │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Ollama (Local LLM)                    │   │
│  │                    Qwen 2.5 7B Model                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Container Configuration

| Container | Image | Ports | Volumes |
|-----------|-------|-------|---------|
| tg4-app | python:3.10 | - | ./:/app |
| timescaledb | timescale/timescaledb | 5432 | pgdata |
| redis | redis:alpine | 6379 | - |
| prometheus | prom/prometheus | 9090 | ./prometheus.yml |
| grafana | grafana/grafana | 3000 | grafana-data |

## Data Volumes

| Volume | Purpose | Location |
|--------|---------|----------|
| pgdata | TimescaleDB data | /var/lib/postgresql/data |
| grafana-data | Grafana dashboards | /var/lib/grafana |
| ./data | Application data | /app/data |

## External Connections

| Service | Protocol | Port | Purpose |
|---------|----------|------|---------|
| Kraken | WSS | 443 | Market data, trading |
| Claude API | HTTPS | 443 | LLM decisions |
| Grok API | HTTPS | 443 | LLM decisions |
| GPT-4 API | HTTPS | 443 | LLM decisions |
