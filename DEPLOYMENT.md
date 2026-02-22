# 🚀 Free Deployment Guide

## Choose Your Free Hosting Platform

This guide covers **4 FREE** deployment options for your Fake News Detection app.

---

## 🌐 Option 1: Streamlit Community Cloud (Recommended - Easiest)

**Cost**: 100% FREE  
**Requirements**: GitHub account  
**RAM**: 1 GB  
**CPU**: 2 cores  
**Perfect for**: Streamlit apps

### Steps:

1. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Fake News Detection"
   git remote add origin https://github.com/shivamyadav039/FakeNewsDetection.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository: `shivamyadav039/FakeNewsDetection`
   - Main file path: `app.py`
   - Click "Deploy!"

3. **Your app will be live at:**
   ```
   https://shivamyadav039-fakenewsdetection-app-xyz123.streamlit.app
   ```

### ⚠️ Important Notes:
- Models must be committed to repo (they are in `models/` already ✅)
- BERT model too large (>500MB) - exclude it or use `.slugignore`
- App will sleep after inactivity, wakes up automatically

---

## 🔷 Option 2: Hugging Face Spaces

**Cost**: 100% FREE  
**Requirements**: Hugging Face account  
**RAM**: 16 GB (free tier!)  
**CPU**: 2 cores (free), GPU available (paid)  
**Perfect for**: ML/AI apps with large models

### Steps:

1. **Create account at** [huggingface.co](https://huggingface.co)

2. **Create new Space**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Name: `fake-news-detection`
   - SDK: `Streamlit`
   - Click "Create Space"

3. **Upload files via web or git**
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/fake-news-detection
   cd fake-news-detection
   
   # Copy all files
   cp -r /path/to/FakeNewsDetection/* .
   
   # Commit and push
   git add .
   git commit -m "Add Fake News Detection app"
   git push
   ```

4. **Your app will be live at:**
   ```
   https://huggingface.co/spaces/<your-username>/fake-news-detection
   ```

### ✅ Advantages:
- ✅ 16 GB RAM (can handle BERT!)
- ✅ Supports large models
- ✅ Persistent storage
- ✅ Fast deployment
- ✅ Community engagement

---

## 🟣 Option 3: Render (Good for Docker)

**Cost**: FREE tier available  
**Requirements**: GitHub account  
**RAM**: 512 MB (free tier)  
**Perfect for**: Docker deployments

### Steps:

1. **Create account at** [render.com](https://render.com)

2. **Create Web Service**
   - Dashboard → "New +" → "Web Service"
   - Connect GitHub repository
   - Name: `fake-news-detection`
   - Environment: `Docker`
   - Plan: `Free`

3. **Configure**
   - Render auto-detects Dockerfile
   - Click "Create Web Service"

4. **Your app will be live at:**
   ```
   https://fake-news-detection.onrender.com
   ```

### ⚠️ Limitations:
- Free tier spins down after 15 min inactivity
- Slower cold starts (~30 seconds)
- 512 MB RAM (may struggle with all models)

---

## 🔵 Option 4: Railway (Developer-Friendly)

**Cost**: $5 FREE credit/month  
**Requirements**: GitHub account  
**RAM**: 512 MB - 8 GB  
**Perfect for**: Multiple services

### Steps:

1. **Create account at** [railway.app](https://railway.app)

2. **Deploy from GitHub**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose `FakeNewsDetection`
   - Railway auto-detects and deploys

3. **Configure**
   - Add custom domain (optional)
   - Environment variables if needed

4. **Your app will be live at:**
   ```
   https://fakenewsdetection-production.up.railway.app
   ```

### ✅ Advantages:
- Fast deployments
- Auto SSL certificates
- Good free tier
- Easy scaling

---

## 📦 Option 5: Google Cloud Run (Advanced)

**Cost**: FREE tier: 2M requests/month  
**Requirements**: Google Cloud account  
**RAM**: Up to 4 GB  
**Perfect for**: Scalable production apps

### Steps:

1. **Install Google Cloud SDK**
   ```bash
   # Install gcloud CLI
   brew install --cask google-cloud-sdk  # macOS
   ```

2. **Initialize and deploy**
   ```bash
   # Login
   gcloud auth login
   
   # Set project
   gcloud config set project YOUR_PROJECT_ID
   
   # Deploy
   gcloud run deploy fake-news-detection \
     --source . \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

3. **Your app will be live at:**
   ```
   https://fake-news-detection-xyz123-uc.a.run.app
   ```

---

## 🎯 Quick Comparison

| Platform | FREE RAM | Models Supported | Setup Time | Recommended For |
|----------|----------|------------------|------------|-----------------|
| **Streamlit Cloud** | 1 GB | Small-Medium | ⚡ 5 min | **Streamlit apps** ⭐ |
| **Hugging Face** | 16 GB | All (incl. BERT) | ⚡ 10 min | **ML/AI apps** ⭐⭐⭐ |
| **Render** | 512 MB | Small only | ⚡ 10 min | Docker apps |
| **Railway** | Variable | Medium | ⚡ 5 min | Multi-service |
| **Cloud Run** | 4 GB | Medium-Large | ⏱️ 20 min | Production scale |

---

## 🏆 Recommended Approach

### Best Option: **Hugging Face Spaces** 🥇

**Why?**
- ✅ 16 GB RAM (handles all models including BERT!)
- ✅ Perfect for ML/AI applications
- ✅ Free forever
- ✅ Great community
- ✅ Persistent storage
- ✅ Easy to set up

### Fallback: **Streamlit Cloud** 🥈

**Why?**
- ✅ Built specifically for Streamlit
- ✅ Super easy deployment (3 clicks)
- ✅ Free forever
- ⚠️ 1 GB RAM (exclude BERT model)

---

## 📝 Pre-Deployment Checklist

### ✅ Required Files (Already Created):
- [x] `requirements.txt` - Python dependencies
- [x] `Dockerfile` - Docker configuration
- [x] `Procfile` - Heroku/Render config
- [x] `setup.sh` - Streamlit config script
- [x] `runtime.txt` - Python version
- [x] `.slugignore` - Files to exclude
- [x] `pyproject.toml` - Project metadata

### ⚠️ Model Size Optimization:

**Problem**: Some platforms have size limits (e.g., 500 MB)

**Solutions**:

1. **Exclude BERT** (use only ensemble)
   ```bash
   # Add to .slugignore or .dockerignore
   models/bert_model/
   ```

2. **Use model compression** (optional)
   ```python
   # Compress models with joblib compression
   joblib.dump(model, 'model.pkl', compress=3)
   ```

3. **Download models on startup** (advanced)
   ```python
   # Download from cloud storage on first run
   # Store in Google Drive, S3, etc.
   ```

---

## 🚀 Quick Deploy Commands

### For Streamlit Cloud:
```bash
# 1. Push to GitHub
git init
git add .
git commit -m "Deploy Fake News Detection"
git remote add origin https://github.com/shivamyadav039/FakeNewsDetection.git
git push -u origin main

# 2. Go to share.streamlit.io and deploy
```

### For Hugging Face Spaces:
```bash
# 1. Clone your HF Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/fake-news-detection
cd fake-news-detection

# 2. Copy files
cp -r ../FakeNewsDetection/* .

# 3. Push
git add .
git commit -m "Initial deployment"
git push
```

### For Docker-based (Render/Railway):
```bash
# Just connect GitHub repo - auto-deploys!
```

---

## 🎨 Custom Domain (Optional)

All platforms support custom domains:

**Streamlit Cloud**: Settings → Custom domain  
**Hugging Face**: Not directly, use proxy  
**Render**: Settings → Custom domain (FREE)  
**Railway**: Settings → Domains (FREE)  

---

## 📊 After Deployment

### Test Your Deployment:

1. **Visit your app URL**
2. **Test all models** (TF-IDF, RF, XGBoost, Ensemble)
3. **Check performance** (response time)
4. **Monitor logs** for errors

### Share Your App:

```
🎉 My Fake News Detection App is live!

🔗 Try it: [YOUR_APP_URL]

✨ Features:
- 99.82% accuracy ensemble model
- Real-time predictions
- Multiple AI models (TF-IDF, Random Forest, XGBoost, Ensemble)
- Beautiful Streamlit UI

#AI #MachineLearning #FakeNews #DataScience
```

---

## 🆘 Troubleshooting

### Issue: App won't start
**Solution**: Check logs for missing dependencies

### Issue: Models not loading
**Solution**: Ensure `models/` folder is committed to repo

### Issue: Out of memory
**Solution**: Exclude large models (BERT) or upgrade plan

### Issue: Slow loading
**Solution**: Use model caching (`@st.cache_resource`)

---

## 💡 Pro Tips

1. **Use `.slugignore`** to reduce deploy size
2. **Enable caching** in Streamlit (`@st.cache_resource`)
3. **Monitor usage** to avoid hitting limits
4. **Add README.md** to repo for better presentation
5. **Use environment variables** for sensitive data

---

## 📞 Support

**Streamlit Cloud**: [docs.streamlit.io/streamlit-community-cloud](https://docs.streamlit.io/streamlit-community-cloud)  
**Hugging Face**: [huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)  
**Render**: [render.com/docs](https://render.com/docs)  
**Railway**: [docs.railway.app](https://docs.railway.app)

---

## 🎯 Next Steps

1. Choose your platform (Hugging Face Spaces recommended!)
2. Follow the deployment steps above
3. Share your live app URL
4. Celebrate! 🎉

---

**Generated**: February 22, 2026  
**Your app is ready to deploy for FREE!** 🚀
