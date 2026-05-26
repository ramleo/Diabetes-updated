# Docker Guide — Diabetes Prediction API

## Image Details

| Property   | Value                  |
|------------|------------------------|
| Base image | python:3.11-slim       |
| Port       | 8000                   |
| User       | appuser (non-root)     |
| Workdir    | /app                   |
| Image name | diabetes-updated:latest|
| Image size | 862 MB                 |

---

## Build Command

```bash
docker build -t diabetes-updated:latest .
```

---

## Run Command

```bash
docker run -d -p 8000:8000 --name diabetes-api diabetes-updated:latest
```

---

## Curl Examples (local)

### Health check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status":"ok","model":"LogisticRegression"}
```

### Single prediction

```bash
curl -X POST http://localhost:8000/predict \
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

Expected response:
```json
{
  "prediction": 1,
  "label": "diabetes",
  "probability_no_diabetes": 0.2941,
  "probability_diabetes": 0.7059
}
```

### Batch prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
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
  ]'
```

Expected response:
```json
{
  "predictions": [
    {"prediction": 1, "label": "diabetes", "probability_no_diabetes": 0.2941, "probability_diabetes": 0.7059},
    {"prediction": 0, "label": "no diabetes", "probability_no_diabetes": 0.965, "probability_diabetes": 0.035}
  ],
  "count": 2
}
```

### Swagger UI

Open in browser:
```
http://localhost:8000/docs
```

---

## Post-Deploy Test Commands (replace `localhost:8000` with your live URL)

```bash
# Health check
curl https://<your-live-url>/health

# Single prediction
curl -X POST https://<your-live-url>/predict \
  -H "Content-Type: application/json" \
  -d '{"Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35, "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50}'

# Batch prediction
curl -X POST https://<your-live-url>/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{"Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35, "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50}]'
```

---

## Useful Docker Commands Reference

| Command | Description |
|---------|-------------|
| `docker build -t diabetes-updated:latest .` | Build image from Dockerfile |
| `docker run -d -p 8000:8000 --name diabetes-api diabetes-updated:latest` | Run container in background |
| `docker stop diabetes-api` | Stop the running container |
| `docker rm diabetes-api` | Remove the stopped container |
| `docker logs diabetes-api` | View container logs |
| `docker logs -f diabetes-api` | Follow (tail) container logs |
| `docker exec -it diabetes-api /bin/bash` | Open shell inside running container |
| `docker ps` | List running containers |
| `docker ps -a` | List all containers (including stopped) |
| `docker images` | List all local images |
| `docker rmi diabetes-updated:latest` | Remove the image |
| `docker system prune` | Remove all stopped containers and dangling images |
