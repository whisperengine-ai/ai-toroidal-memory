# Docker Deployment Guide

> 📚 **Documentation:** [README](README.md) | [API Docs](docs/API_DOCUMENTATION.md) | [Test Results](docs/DOCKER_TEST_RESULTS.md) | [Terminal Chat](docs/TERMINAL_CHAT.md)

Run the Toroidal Memory API server in Docker with persistent storage.

## Quick Start

```bash
# Build and start the server
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the server
docker-compose down

# Stop and remove data
docker-compose down -v
```

The API will be available at `http://localhost:3000`

## Directory Structure

```
.
├── docker-compose.yml    # Docker Compose configuration
├── Dockerfile           # Container build instructions
└── data/               # Persistent storage (created automatically)
    └── *.json          # Memory state files (JSON format)
```

**Note:** The API currently uses **JSON format** for file persistence. The library also supports binary and compressed formats (see `examples/persistence.rs`), but the REST API uses JSON for human readability and portability.

## Configuration

### Environment Variables

Configure the server via environment variables in `docker-compose.yml`:

```yaml
environment:
  - PORT=3000              # Server port
  - STORAGE_PATH=/data     # Storage directory inside container
  - RUST_LOG=info         # Log level (error, warn, info, debug, trace)
```

### Port Mapping

Change the exposed port in `docker-compose.yml`:

```yaml
ports:
  - "8080:3000"  # Expose on port 8080 instead of 3000
```

### Storage Volume

The `data/` directory is mapped to `/data` inside the container. All memory files are persisted here.

```yaml
volumes:
  - toroidal-data:/data
```

## Usage Examples

### Create a Memory Instance

```bash
curl -X POST http://localhost:3000/api/v1/memories \
  -H 'Content-Type: application/json' \
  -d '{
    "width": 100,
    "height": 100,
    "diffusion_config": {
      "diffusion_rate": 0.25,
      "decay_rate": 0.1,
      "threshold": 0.01
    }
  }'
```

Response:
```json
{
  "memory_id": "mem_abc123",
  "width": 100,
  "height": 100,
  "message": "Memory mem_abc123 created successfully"
}
```

### Set Cell Values

```bash
curl -X POST http://localhost:3000/api/v1/memories/mem_abc123/cell \
  -H 'Content-Type: application/json' \
  -d '{"x": 50, "y": 50, "value": 1.0}'
```

### Run Diffusion

```bash
curl -X POST http://localhost:3000/api/v1/memories/mem_abc123/diffusion \
  -H 'Content-Type: application/json' \
  -d '{"steps": 10}'
```

### Save to File

```bash
curl -X POST http://localhost:3000/api/v1/memories/mem_abc123/save \
  -H 'Content-Type: application/json' \
  -d '{"filename": "my_memory.json"}'
```

This saves to `./data/my_memory.json` on the host.

### Load from File

```bash
curl -X POST http://localhost:3000/api/v1/memories/mem_abc123/load \
  -H 'Content-Type: application/json' \
  -d '{"filename": "my_memory.json"}'
```

### Get Statistics

```bash
curl http://localhost:3000/api/v1/memories/mem_abc123/stats
```

### Health Check

```bash
curl http://localhost:3000/health
```

## Building

### Build Image Manually

```bash
docker build -t toroidal-memory-api .
```

### Run Without Compose

```bash
docker run -d \
  --name toroidal-memory \
  -p 3000:3000 \
  -v $(pwd)/data:/data \
  -e STORAGE_PATH=/data \
  toroidal-memory-api
```

## Troubleshooting

### View Logs

```bash
# Real-time logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100
```

### Container Not Starting

Check logs for errors:
```bash
docker-compose logs toroidal-memory-api
```

### Permission Issues

Ensure the `data/` directory is writable:
```bash
mkdir -p data
chmod 777 data
```

### Health Check Failing

Install curl in the container (already included in Dockerfile) or disable health check in `docker-compose.yml`.

## Performance

### Multi-Core Support

The server uses Rayon for parallel processing. Limit CPU cores:

```yaml
services:
  toroidal-memory-api:
    cpus: 4  # Limit to 4 cores
```

### Memory Limits

Set memory limits:

```yaml
services:
  toroidal-memory-api:
    mem_limit: 2g
    mem_reservation: 1g
```

## Production Deployment

### Use Pre-built Image

Tag and push to a registry:

```bash
docker build -t myregistry/toroidal-memory:latest .
docker push myregistry/toroidal-memory:latest
```

### Run in Production

```yaml
version: '3.8'

services:
  toroidal-memory-api:
    image: myregistry/toroidal-memory:latest
    restart: always
    ports:
      - "3000:3000"
    volumes:
      - /mnt/storage/toroidal:/data
    environment:
      - RUST_LOG=warn
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 2G
        reservations:
          cpus: '2'
          memory: 1G
```

### Reverse Proxy (nginx)

```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Scaling

### Multiple Instances

Each instance needs its own storage:

```yaml
services:
  api-1:
    extends: toroidal-memory-api
    volumes:
      - ./data1:/data
    ports:
      - "3001:3000"

  api-2:
    extends: toroidal-memory-api
    volumes:
      - ./data2:/data
    ports:
      - "3002:3000"
```

Use a load balancer (nginx, HAProxy) to distribute requests.

### Shared Storage

For shared memory state, use networked storage (NFS, S3):

```yaml
volumes:
  toroidal-data:
    driver: local
    driver_opts:
      type: nfs
      o: addr=nfs-server,rw
      device: ":/path/to/share"
```

## Backup

### Manual Backup

```bash
# Backup data directory
tar -czf backup-$(date +%Y%m%d).tar.gz data/

# Restore
tar -xzf backup-20250101.tar.gz
```

### Automated Backup

Add a backup service to `docker-compose.yml`:

```yaml
services:
  backup:
    image: alpine:latest
    volumes:
      - toroidal-data:/data:ro
      - ./backups:/backups
    command: |
      sh -c "while true; do
        tar -czf /backups/backup-$(date +%Y%m%d-%H%M%S).tar.gz /data
        sleep 86400
      done"
```

## Monitoring

### Prometheus Metrics (Future)

Expose metrics endpoint:

```bash
curl http://localhost:3000/metrics
```

### Logging

Aggregate logs with Docker logging driver:

```yaml
services:
  toroidal-memory-api:
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
```

## API Documentation

Full API reference: [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md)

## Support

- Issues: GitHub Issues
- Documentation: `/docs` directory
- Examples: `/examples` directory
