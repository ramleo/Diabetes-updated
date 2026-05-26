# Deployment Guide — diabetes-updated

## 1. Prerequisites

- **Docker** installed and running locally (for local testing)
- **Render account** at [https://render.com](https://render.com)
- **GitHub repo** connected: `https://github.com/ramleo/Diabetes-updated`

---

## 2. Render Deployment (Recommended)

### Option A — Render Blueprints (automatic via `render.yaml`)

1. Go to [https://dashboard.render.com](https://dashboard.render.com)
2. Click **New → Blueprint**
3. Connect your GitHub repo: `ramleo/Diabetes-updated`
4. Render will detect `render.yaml` and auto-configure the service
5. Click **Apply** to deploy

> `render.yaml` auto-configures the service name (`diabetes-updated`), runtime (Docker), region (Oregon), health check path (`/health`), and port (`8000`).

### Option B — Manual Web Service Setup

1. Go to [https://dashboard.render.com](https://dashboard.render.com)
2. Click **New → Web Service**
3. Connect your GitHub repo: `ramleo/Diabetes-updated`
4. Configure the service:
   - **Name:** `diabetes-updated`
   - **Runtime:** Docker
   - **Dockerfile Path:** `./Dockerfile`
   - **Docker Context:** `.`
   - **Plan:** Free
   - **Region:** Oregon (US West)
5. Under **Environment Variables**, add:
   - `PORT` = `8000`
6. Click **Create Web Service**

Render will build the Docker image and deploy automatically. The first deploy typically takes 3–5 minutes.

---

## 3. Live API Endpoints

Base URL: `https://diabetes-updated.onrender.com`

| Method | Endpoint         | Description                          |
|--------|-----------------|--------------------------------------|
| GET    | `/`             | Redirects to `/docs` (Swagger UI)    |
| GET    | `/health`       | Health check — returns service status |
| POST   | `/predict`      | Single-sample diabetes prediction    |
| POST   | `/predict/batch`| Batch predictions (multiple samples) |

---

## 4. Test the Live API

Replace `localhost:8000` with `https://diabetes-updated.onrender.com` for live testing.

### Health Check
```bash
curl https://diabetes-updated.onrender.com/health
```

### Single Prediction
```bash
curl -X POST https://diabetes-updated.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
  }'
```

### Batch Prediction
```bash
curl -X POST https://diabetes-updated.onrender.com/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
      },
      {
        "Pregnancies": 1,
        "Glucose": 85,
        "BloodPressure": 66,
        "SkinThickness": 29,
        "Insulin": 0,
        "BMI": 26.6,
        "DiabetesPedigreeFunction": 0.351,
        "Age": 31
      }
    ]
  }'
```

### Swagger UI
Open in browser:
```
https://diabetes-updated.onrender.com/docs
```

---

## 5. Re-deploy

- **Automatic:** Every push to `main` triggers an automatic re-deploy (controlled by `autoDeploy: true` in `render.yaml`)
- **Manual:** Go to Render dashboard → select `diabetes-updated` → click **Manual Deploy → Deploy latest commit**

---

## 6. Environment Variables

| Variable | Value | Description                        |
|----------|-------|------------------------------------|
| `PORT`   | `8000`| Port the FastAPI server listens on |

---

## 7. Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| **Cold start delay (~30s)** | Free tier spins down after inactivity | First request after idle period will be slow; subsequent requests are fast |
| **Health check failing** | Wrong health check path | Must be `/health` — already set in `render.yaml` |
| **Docker build error** | Missing dependency or syntax issue | Check Render build logs; ensure `requirements.txt` is up to date |
| **502 Bad Gateway** | App still starting up | Wait 30–60 seconds and retry; check logs for startup errors |
| **Model not found error** | `models/` not copied into Docker image | Verify `Dockerfile` COPY includes `models/`; check `.dockerignore` excludes only non-essential files |
| **Port binding error** | App not reading `PORT` env var | Ensure `app.py` uses `os.getenv("PORT", 8000)` in the uvicorn startup |
