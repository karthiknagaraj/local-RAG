# Local RAG - Deployment Guide

Complete guide for deploying the Local RAG application in production environments.

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Deployment Scenarios](#deployment-scenarios)
5. [Security Hardening](#security-hardening)
6. [Monitoring & Maintenance](#monitoring--maintenance)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 11+
- **CPU**: 4+ cores (Intel Core i5/AMD Ryzen 5 or better)
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **Python**: 3.9 - 3.13

### Recommended Requirements (Production)
- **CPU**: 8+ cores (Intel Xeon/AMD EPYC)
- **RAM**: 32GB
- **Storage**: 50GB SSD
- **Python**: 3.11 (best compatibility)
- **Network**: 1Gbps LAN (for multi-user deployments)

---

## Installation

### Step 1: System Preparation

#### Windows
```powershell
# Enable long paths (Admin PowerShell)
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# Install Visual C++ Redistributables (for llama-cpp-python)
# Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
```

#### Linux (Ubuntu/Debian)
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install build essentials
sudo apt install -y build-essential python3-dev python3-pip python3-venv

# Install system dependencies
sudo apt install -y libgomp1
```

#### macOS
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11
brew install python@3.11
```

---

### Step 2: Clone/Download Project

```bash
# Option 1: Git clone (if version controlled)
git clone https://github.com/your-org/local-rag.git
cd local-rag

# Option 2: Extract from zip
unzip local_rag.zip
cd local_rag
```

---

### Step 3: Virtual Environment Setup

#### Windows
```powershell
# Create virtual environment
python -m venv .venv

# Activate
.\.venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

#### Linux/macOS
```bash
# Create virtual environment
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

---

### Step 4: Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# Verify installation
pip list | grep -E "(sentence-transformers|faiss-cpu|llama-cpp-python|gradio)"
```

**Expected Output:**
```
faiss-cpu               1.13.1
gradio                  6.1.0
llama-cpp-python        0.3.2
sentence-transformers   3.2.1
```

---

### Step 5: Download Models

#### Embedding Model (Auto-downloaded on first run)
```bash
# Trigger download manually (optional)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

#### LLM Model

**Option A: Recommended (Qwen 2.5 3B)**
```bash
# Using Python script
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Qwen/Qwen2.5-3B-Instruct-GGUF', filename='qwen2.5-3b-instruct-q4_k_m.gguf', local_dir='local_rag/models')"
```

**Option B: Manual Download**
1. Visit: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF
2. Download: `qwen2.5-3b-instruct-q4_k_m.gguf` (2.1GB)
3. Move to: `local_rag/models/`

---

### Step 6: Configuration

Edit `local_rag/config.yaml`:

```yaml
docs_dir: "/path/to/your/documents"  # Update this
index_dir: "/path/to/local_rag/index"
model_path: "/path/to/local_rag/models/qwen2.5-3b-instruct-q4_k_m.gguf"
embedding_model: "all-MiniLM-L6-v2"

top_k: 5
chunk_size: 1200
chunk_overlap: 200

llm:
  n_ctx: 4096
  n_threads: null  # Auto-detect CPU cores
  n_gpu_layers: 0  # 0 = CPU-only
  temperature: 0.2
  max_tokens: 512
```

**Important:** Use absolute paths for `docs_dir`, `index_dir`, and `model_path`.

---

### Step 7: Build Index

```bash
# Activate virtual environment first
.venv/Scripts/activate  # Windows
source .venv/bin/activate  # Linux/macOS

# Build index from documents
python -m local_rag.src.ingest --config local_rag/config.yaml
```

**Expected Output:**
```
Building index from: /path/to/documents
Found 25 files
Processed 342 chunks from 25 files
Index saved to: /path/to/index
```

---

### Step 8: Test Installation

```bash
# CLI test
python -m local_rag.src.query --config local_rag/config.yaml --q "What are the main topics in these documents?"

# Web UI test (in background)
python -m local_rag.src.web_ui --config local_rag/config.yaml
# Open browser to: http://127.0.0.1:7860
```

---

## Deployment Scenarios

### Scenario 1: Single-User Desktop (Development)

**Use Case:** Personal knowledge base, local development

**Configuration:**
```yaml
# config.yaml
llm:
  n_ctx: 4096
  n_threads: 4  # Limit to avoid system slowdown
  temperature: 0.2
```

**Launch:**
```bash
# CLI mode
python -m local_rag.src.query --config config.yaml --repl

# Web UI mode (local only)
python -m local_rag.src.web_ui --config config.yaml
```

---

### Scenario 2: Team Server (Internal Network)

**Use Case:** 5-20 users, internal company network

**Architecture:**
```
[Users] --> [Nginx Reverse Proxy] --> [Gradio Web UI]
                                         |
                                    [FAISS Index]
                                    [LLM (CPU)]
```

**Server Setup:**

1. **Install Nginx**
```bash
# Ubuntu
sudo apt install nginx

# Configure /etc/nginx/sites-available/local-rag
upstream rag_backend {
    server 127.0.0.1:7860;
}

server {
    listen 80;
    server_name rag.company.internal;
    
    location / {
        proxy_pass http://rag_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

2. **Enable Nginx config**
```bash
sudo ln -s /etc/nginx/sites-available/local-rag /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

3. **Run as systemd service**

Create `/etc/systemd/system/local-rag.service`:
```ini
[Unit]
Description=Local RAG Web UI
After=network.target

[Service]
Type=simple
User=raguser
WorkingDirectory=/opt/local-rag
Environment="PATH=/opt/local-rag/.venv/bin"
ExecStart=/opt/local-rag/.venv/bin/python -m local_rag.src.web_ui --config /opt/local-rag/config.yaml --port 7860
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

4. **Enable and start service**
```bash
sudo systemctl daemon-reload
sudo systemctl enable local-rag
sudo systemctl start local-rag
sudo systemctl status local-rag
```

---

### Scenario 3: High-Availability Production

**Use Case:** 50+ users, mission-critical

**Architecture:**
```
                [Load Balancer (HAProxy)]
                /           |           \
        [App Instance 1] [Instance 2] [Instance 3]
                \           |           /
                    [Shared NFS]
                   /           \
            [FAISS Index]  [Models]
```

**HAProxy Configuration:**
```
# /etc/haproxy/haproxy.cfg
frontend rag_frontend
    bind *:80
    default_backend rag_backend

backend rag_backend
    balance roundrobin
    option httpchk GET /
    server rag1 192.168.1.101:7860 check
    server rag2 192.168.1.102:7860 check
    server rag3 192.168.1.103:7860 check
```

**NFS Shared Storage:**
```bash
# Server (192.168.1.100)
sudo apt install nfs-kernel-server
sudo mkdir -p /mnt/rag-data
sudo chown nobody:nogroup /mnt/rag-data

# /etc/exports
/mnt/rag-data 192.168.1.0/24(rw,sync,no_subtree_check)

sudo exportfs -a
sudo systemctl restart nfs-kernel-server

# Clients (each instance)
sudo apt install nfs-common
sudo mount 192.168.1.100:/mnt/rag-data /opt/local-rag/shared
```

---

## Security Hardening

### 1. Network Security

**Firewall Rules (iptables):**
```bash
# Allow only internal network
sudo iptables -A INPUT -p tcp --dport 7860 -s 192.168.1.0/24 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 7860 -j DROP

# Persist rules
sudo apt install iptables-persistent
sudo netfilter-persistent save
```

**Nginx HTTPS (SSL/TLS):**
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d rag.company.com

# Auto-renewal
sudo systemctl enable certbot.timer
```

---

### 2. Application Security

**Environment Variables (Sensitive Config):**
```bash
# .env file (never commit to git)
MODEL_PATH=/secure/path/to/models/qwen.gguf
DOCS_DIR=/secure/path/to/docs
INDEX_DIR=/secure/path/to/index
```

**Load in Python:**
```python
# config_loader.py
import os
from dotenv import load_dotenv

load_dotenv()

config = {
    "model_path": os.getenv("MODEL_PATH"),
    "docs_dir": os.getenv("DOCS_DIR"),
    # ...
}
```

---

### 3. Access Control

**Basic Auth via Nginx:**
```bash
# Create password file
sudo apt install apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd username

# Update nginx config
location / {
    auth_basic "Restricted Access";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://rag_backend;
}
```

**Gradio Built-in Auth:**
```python
# web_ui.py
interface.launch(
    auth=[("admin", "secure_password")],
    auth_message="Enter credentials"
)
```

---

### 4. File Permissions

```bash
# Restrict config file access
chmod 600 config.yaml

# Restrict model directory
chmod 700 local_rag/models
chown raguser:raguser local_rag/models

# Restrict index directory
chmod 700 local_rag/index
```

---

## Monitoring & Maintenance

### Health Check Endpoint

Add to `web_ui.py`:
```python
import gradio as gr

def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Add to interface
gr.Interface(fn=health_check, inputs=None, outputs="json")
```

---

### Logging Configuration

```python
# logging_config.py
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
handler = RotatingFileHandler(
    'logs/local_rag.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)

logger = logging.getLogger('local_rag')
logger.setLevel(logging.INFO)
logger.addHandler(handler)
```

**Log Analysis:**
```bash
# View recent errors
tail -f logs/local_rag.log | grep ERROR

# Count queries per hour
grep "answer_question" logs/local_rag.log | cut -d' ' -f1-2 | uniq -c
```

---

### Performance Monitoring

**System Metrics (Prometheus + Grafana):**

1. Install Prometheus client
```bash
pip install prometheus-client
```

2. Add metrics to `web_ui.py`
```python
from prometheus_client import Counter, Histogram, start_http_server

# Metrics
query_counter = Counter('rag_queries_total', 'Total queries')
query_duration = Histogram('rag_query_duration_seconds', 'Query duration')

@query_duration.time()
def answer(message, history):
    query_counter.inc()
    # ... existing code
```

3. Start metrics server
```python
start_http_server(8000)  # Prometheus scrapes :8000/metrics
```

---

### Backup Strategy

**Automated Backup Script:**
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/local-rag"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup index
tar -czf "$BACKUP_DIR/index_$DATE.tar.gz" /opt/local-rag/index

# Backup config
cp /opt/local-rag/config.yaml "$BACKUP_DIR/config_$DATE.yaml"

# Cleanup old backups (keep last 30 days)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +30 -delete
```

**Cron Schedule:**
```bash
# Backup daily at 2 AM
0 2 * * * /opt/local-rag/backup.sh
```

---

### Update Procedure

**1. Update Code:**
```bash
cd /opt/local-rag
git pull origin main
```

**2. Update Dependencies:**
```bash
source .venv/bin/activate
pip install -r requirements.txt --upgrade
```

**3. Rebuild Index (if schema changed):**
```bash
python -m local_rag.src.ingest --config config.yaml
```

**4. Restart Service:**
```bash
sudo systemctl restart local-rag
sudo systemctl status local-rag
```

---

## Troubleshooting

### Issue: Service won't start

**Check logs:**
```bash
sudo journalctl -u local-rag -n 50
```

**Common causes:**
- Wrong file paths in config.yaml
- Missing models
- Port already in use

**Solutions:**
```bash
# Check port usage
sudo netstat -tlnp | grep 7860

# Kill conflicting process
sudo kill -9 <PID>

# Verify config paths
python -c "from local_rag.src.rag import load_config; print(load_config('config.yaml'))"
```

---

### Issue: High memory usage

**Diagnosis:**
```bash
# Monitor memory
htop

# Check Python process
ps aux | grep python | awk '{print $6/1024 "MB - " $11}'
```

**Solutions:**
1. Reduce `n_ctx` in config (4096 → 2048)
2. Use smaller model (TinyLlama 1.1B instead of Qwen 3B)
3. Reduce `top_k` (5 → 3)

---

### Issue: Slow query responses

**Diagnosis:**
```bash
# Profile query
python -m cProfile -o profile.stats -m local_rag.src.query --config config.yaml --q "test"

# Analyze
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumtime'); p.print_stats(10)"
```

**Bottlenecks:**
- LLM inference: 80% of time (expected)
- FAISS search: <5% (acceptable)
- Embedding: <10% (acceptable)

**Solutions:**
1. Use GPU offloading (`n_gpu_layers: 35`)
2. Reduce `max_tokens` (512 → 256)
3. Use smaller model

---

### Issue: Index corruption

**Symptoms:**
- FAISS errors
- Dimension mismatches
- Missing metadata

**Recovery:**
```bash
# Backup corrupted index
mv local_rag/index local_rag/index_corrupt_$(date +%Y%m%d)

# Rebuild from scratch
python -m local_rag.src.ingest --config config.yaml
```

---

## Production Checklist

### Pre-Deployment
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip list`)
- [ ] Models downloaded and verified
- [ ] Config file uses absolute paths
- [ ] Index built successfully
- [ ] CLI test passed
- [ ] Web UI test passed
- [ ] File permissions set (600 for config, 700 for models/index)
- [ ] Firewall rules configured
- [ ] Backup strategy implemented

### Post-Deployment
- [ ] Service running (`systemctl status local-rag`)
- [ ] Health check endpoint responding
- [ ] Logs rotating properly
- [ ] Monitoring alerts configured
- [ ] SSL certificate valid (if using HTTPS)
- [ ] Access control working
- [ ] Performance metrics baseline established

### Ongoing Maintenance
- [ ] Weekly: Review error logs
- [ ] Monthly: Update dependencies
- [ ] Quarterly: Rebuild index (if documents changed)
- [ ] Annually: Review and update security policies

---

## Performance Benchmarks

### Expected Latency (per query)

| Component | CPU-only | GPU (RTX 3080) |
|-----------|----------|----------------|
| Embedding | 50ms | 20ms |
| FAISS Search (10k chunks) | 30ms | 30ms |
| Context Building | 5ms | 5ms |
| LLM Generation (Qwen 3B) | 8s | 1.5s |
| **Total** | **~8.1s** | **~1.6s** |

### Throughput

| Setup | Concurrent Users | Queries/Min |
|-------|------------------|-------------|
| Single instance (CPU) | 1 | 7 |
| Single instance (GPU) | 5 | 30 |
| 3x instances (CPU) | 10 | 20 |
| 3x instances (GPU) | 20 | 90 |

---

## Support & Resources

### Documentation
- [Full Documentation](DOCUMENTATION.md)
- [API Reference](API_REFERENCE.md)
- [Architecture](ARCHITECTURE.md)

### Dependencies
- [Sentence Transformers Docs](https://www.sbert.net/)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [Gradio Docs](https://www.gradio.app/docs)

---

**Last Updated:** December 16, 2025
**Version:** 1.0.0
