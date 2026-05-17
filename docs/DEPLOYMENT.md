# 🚀 Power Data AI - Deployment Guide

This guide walks you through deploying Power Data AI to various platforms.

---

## 📋 Prerequisites

- Python 3.8+ installed
- Git installed
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- GitHub account

---

## Option 1: Streamlit Cloud (Recommended - Free)

**Best for:** Quick deployment, no infrastructure management

### Steps:

1. **Push to GitHub**
   ```bash
   cd powerdata-ai
   git init
   git add .
   git commit -m "Initial commit: Power Data AI v1.0.0"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/powerdata-ai.git
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository: `YOUR-USERNAME/powerdata-ai`
   - Branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

3. **Add Secrets**
   - In Streamlit Cloud dashboard, go to app settings
   - Click "Secrets"
   - Add:
     ```toml
     OPENAI_API_KEY = "sk-your-actual-api-key-here"
     ```
   - Save

4. **Your app is live! 🎉**
   - URL will be: `https://YOUR-USERNAME-powerdata-ai.streamlit.app`

---

## Option 2: Docker Deployment

**Best for:** Self-hosting, custom infrastructure

### Steps:

1. **Build Docker Image**
   ```bash
   docker build -t powerdata-ai:latest .
   ```

2. **Run Container**
   ```bash
   docker run -d \
     -p 8501:8501 \
     -e OPENAI_API_KEY="your-api-key-here" \
     --name powerdata-ai \
     powerdata-ai:latest
   ```

3. **Access Application**
   - Open browser to `http://localhost:8501`

4. **Stop Container**
   ```bash
   docker stop powerdata-ai
   docker rm powerdata-ai
   ```

---

## Option 3: Heroku

**Best for:** Production deployment with custom domain

### Steps:

1. **Install Heroku CLI**
   ```bash
   # Mac
   brew tap heroku/brew && brew install heroku
   
   # Windows
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Login to Heroku**
   ```bash
   heroku login
   ```

3. **Create Heroku App**
   ```bash
   heroku create your-app-name
   ```

4. **Add Buildpack**
   ```bash
   heroku buildpacks:set heroku/python
   ```

5. **Set Environment Variables**
   ```bash
   heroku config:set OPENAI_API_KEY="your-api-key-here"
   ```

6. **Create `setup.sh`**
   ```bash
   mkdir -p ~/.streamlit/
   echo "\
   [server]\n\
   headless = true\n\
   port = \$PORT\n\
   enableCORS = false\n\
   \n\
   " > ~/.streamlit/config.toml
   ```

7. **Create `Procfile`**
   ```
   web: sh setup.sh && streamlit run app.py
   ```

8. **Deploy**
   ```bash
   git push heroku main
   ```

9. **Open App**
   ```bash
   heroku open
   ```

---

## Option 4: AWS EC2

**Best for:** Full control, enterprise deployment

### Steps:

1. **Launch EC2 Instance**
   - Ubuntu 22.04 LTS
   - t2.medium or larger
   - Open port 8501 in security group

2. **Connect via SSH**
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   ```

3. **Install Dependencies**
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv git -y
   ```

4. **Clone Repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/powerdata-ai.git
   cd powerdata-ai
   ```

5. **Set Up Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

6. **Set Environment Variables**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

7. **Run with nohup**
   ```bash
   nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
   ```

8. **Access Application**
   - `http://your-instance-ip:8501`

---

## Option 5: Google Cloud Run

**Best for:** Serverless, auto-scaling

### Steps:

1. **Install Google Cloud SDK**
   ```bash
   # Follow: https://cloud.google.com/sdk/docs/install
   ```

2. **Initialize Project**
   ```bash
   gcloud init
   gcloud config set project YOUR-PROJECT-ID
   ```

3. **Build Container**
   ```bash
   gcloud builds submit --tag gcr.io/YOUR-PROJECT-ID/powerdata-ai
   ```

4. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy powerdata-ai \
     --image gcr.io/YOUR-PROJECT-ID/powerdata-ai \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars OPENAI_API_KEY="your-api-key-here"
   ```

5. **Your app is deployed!**
   - URL will be provided in output

---

## 🔒 Security Best Practices

### API Key Management
- ✅ **DO:** Store API keys in environment variables or secrets managers
- ✅ **DO:** Use `.env` files for local development (add to `.gitignore`)
- ❌ **DON'T:** Commit API keys to Git
- ❌ **DON'T:** Hardcode keys in source code

### Production Checklist
- [ ] Enable HTTPS/SSL
- [ ] Set up monitoring and logging
- [ ] Configure rate limiting
- [ ] Enable authentication (if needed)
- [ ] Set up automated backups
- [ ] Configure CDN for static assets
- [ ] Enable error tracking (Sentry, etc.)

---

## 📊 Monitoring

### Streamlit Cloud
- Built-in analytics in dashboard
- View logs in real-time
- Monitor resource usage

### Custom Deployment
Add logging:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
```

---

## 🆘 Troubleshooting

### App won't start
1. Check logs for errors
2. Verify all dependencies installed
3. Confirm API key is set correctly

### Slow performance
1. Upgrade instance size
2. Enable caching in Streamlit
3. Optimize data loading

### API errors
1. Verify OpenAI API key is valid
2. Check account has credits
3. Monitor rate limits

---

## 📞 Support

Need help with deployment?

- 📧 Email: issaka.seogo@seogoglobalimpacts.com
- 🐛 GitHub Issues: [Report a problem](https://github.com/iseogo/powerdata-ai/issues)
- 💬 Discussions: [Ask the community](https://github.com/iseogo/powerdata-ai/discussions)

---

**Power Data AI** – *Turning your data into direction.*

By **Issaka Seogo** | Seogo Global Impact Ltd. Co.
