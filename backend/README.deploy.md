Docker build and deploy instructions

Build locally

```bash
# from repo root
docker build -t thesis-backend -f backend/Dockerfile backend/
docker build -t thesis-backend -f Dockerfile .
docker run --rm -p 8000:8000 thesis-backend
```

Push to registry (example using Docker Hub)

```bash
docker tag thesis-backend risajoy18/thesis-backend:latest
docker push risajoy18/thesis-backend:latest
```

Pull & run the pushed image:

```bash
docker pull risajoy18/thesis-backend:latest
docker run --rm -p 8000:8000 risajoy18/thesis-backend:latest
```

Render deployment options

- Option A (recommended): Deploy as a Docker service on Render. Point Render to the repo and let it build the `backend/Dockerfile`, or push an image to a registry and reference it from Render.
- Option B: Use Render's buildpacks — ensure `runtime.txt` exists with a supported Python (3.11) and hope prebuilt wheels are available for all packages. This is less reliable for geospatial stacks.

Notes
- The image installs system packages required by geospatial libraries (`gdal`, `libgeos`, `libproj`). If you add other native libs, add them to the `apt-get install` list.
- If you prefer a non-root user in the container for security, I can update the Dockerfile to create and use a dedicated user.
