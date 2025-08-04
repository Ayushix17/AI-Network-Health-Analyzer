# 📡 AI Network Health Analyzer Dashboard

A Streamlit-powered dashboard for real-time visualization of network events and anomalies detected by the AI Network Health Analyzer backend.

![Streamlit Screenshot](https://via.placeholder.com/900x300?text=Dashboard+Preview)

---

## 🔍 Overview

This dashboard interacts with a Flask-based AI log analyzer that processes network logs, detects anomalies, and stores metrics in a local SQLite database. The Streamlit frontend provides:

- 📋 Real-time network event viewing
- ⚠️ Anomaly detection insights
- 📊 Summary statistics by event type and metric severity

---

## 🛠 Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: Flask (from `analyzer.py`)
- **Database**: SQLite
- **Monitoring**: Prometheus + InfluxDB (optional)

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-network-health-analyzer.git
cd ai-network-health-analyzer
