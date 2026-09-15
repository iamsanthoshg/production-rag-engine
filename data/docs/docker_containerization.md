# Docker Containerization & Multi-Stage Builds for AI Services

Containerization ensures deterministic environments, reproducible builds, and seamless deployment across development, staging, and production.

## Docker Architecture

### Daemon and Container Runtime
- **Docker Client**: The CLI utility (`docker`) that transmits REST commands to the daemon.
- **dockerd**: The persistent daemon managing images, containers, networks, and storage volumes.
- **containerd & runc**: The low-level Open Container Initiative (OCI) runtime executing container namespaces, cgroups, and rootfs isolation.

## Optimizing Python & AI Dockerfiles

### 1. Multi-Stage Builds
Multi-stage builds decouple the build environment (compilers, headers, heavy wheels) from the final lean runtime image.

```dockerfile
# Build stage
FROM python:3.12-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final runtime stage
FROM python:3.12-slim AS runner
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Layer Caching Strategies
Docker creates read-only layers for every instruction (`RUN`, `COPY`, `ADD`).
- Place infrequently changing instructions (`apt-get install`, `COPY requirements.txt`) near the top.
- Place frequently changing application code (`COPY . .`) at the very end.
- Combine consecutive `RUN apt-get` commands and clean package lists (`rm -rf /var/lib/apt/lists/*`) in a single layer to avoid bloat.

### 3. Storage Drivers and Volumes
- **Bind Mounts**: Maps host directories directly into containers; optimal for hot-reloading during local development.
- **Named Volumes**: Managed entirely by Docker; recommended for persistent state such as PostgreSQL data files or vector databases (`pgvector`, `chroma_db`).
