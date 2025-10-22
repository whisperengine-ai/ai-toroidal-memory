# Toroidal Memory REST API Documentation

> 📚 **Documentation:** [README](../README.md) | [Quick Start](QUICK_REFERENCE.md) | [Terminal Chat](TERMINAL_CHAT.md) | [Docker](../DOCKER.md) | [Implementation Status](IMPLEMENTATION_STATUS.md)

Complete HTTP/JSON API for managing toroidal memory instances.

**Version:** 1.0  
**Base URL:** `http://localhost:3000/api/v1`  
**Format:** JSON

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Endpoints](#endpoints)
4. [Data Models](#data-models)
5. [Examples](#examples)
6. [Error Handling](#error-handling)
7. [Rate Limiting](#rate-limiting)

---

## Overview

This API provides HTTP access to toroidal memory operations including:
- Creating and managing memory instances
- Reading and writing cell values
- Running diffusion simulations
- Pattern operations
- Persistence (save/load)

### Quick Start

```bash
# Create a new memory instance
curl -X POST http://localhost:3000/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"width": 50, "height": 50}'

# Returns: {"memory_id": "mem_abc123", ...}

# Set a cell value
curl -X POST http://localhost:3000/api/v1/memories/mem_abc123/cell \
  -H "Content-Type: application/json" \
  -d '{"x": 25, "y": 25, "value": 1.0}'

# Run diffusion
curl -X POST http://localhost:3000/api/v1/memories/mem_abc123/diffusion \
  -H "Content-Type: application/json" \
  -d '{"steps": 10}'
```

---

## Authentication

**Current Version:** No authentication required (development mode)

**Production:** Add API key in header:
```
Authorization: Bearer YOUR_API_KEY
```

---

## Endpoints

### Memory Management

#### Create Memory Instance

**POST** `/memories`

Creates a new toroidal memory instance.

**Request Body:**
```json
{
  "width": 50,
  "height": 50,
  "diffusion_config": {
    "diffusion_rate": 0.25,
    "decay_rate": 0.1,
    "threshold": 0.01
  },
  "metadata": {
    "name": "My Memory",
    "purpose": "experiment"
  }
}
```

**Response:** `201 Created`
```json
{
  "memory_id": "mem_1729625472abc",
  "width": 50,
  "height": 50,
  "message": "Memory mem_1729625472abc created successfully"
}
```

**Fields:**
- `width` (required): Grid width (1-1000)
- `height` (required): Grid height (1-1000)
- `diffusion_config` (optional): Diffusion parameters
- `metadata` (optional): Key-value pairs for custom data

---

#### List Memories

**GET** `/memories`

Returns all memory instances.

**Response:** `200 OK`
```json
{
  "memories": [
    "mem_1729625472abc",
    "mem_1729625473def"
  ],
  "count": 2
}
```

---

#### Get Memory Info

**GET** `/memories/{memory_id}`

Get information about a specific memory instance.

**Response:** `200 OK`
```json
{
  "memory_id": "mem_1729625472abc",
  "width": 50,
  "height": 50,
  "diffusion_config": {
    "diffusion_rate": 0.25,
    "decay_rate": 0.1,
    "threshold": 0.01
  },
  "metadata": {
    "name": "My Memory"
  }
}
```

---

#### Delete Memory

**DELETE** `/memories/{memory_id}`

Delete a memory instance.

**Response:** `200 OK`
```json
{
  "success": true,
  "message": "Memory mem_1729625472abc deleted"
}
```

---

### Cell Operations

#### Get Cell Value

**GET** `/memories/{memory_id}/cell?x={x}&y={y}`

Get the value of a single cell.

**Query Parameters:**
- `x`: X coordinate (integer, wraps around)
- `y`: Y coordinate (integer, wraps around)

**Response:** `200 OK`
```json
{
  "x": 25,
  "y": 25,
  "value": 0.85
}
```

---

#### Set Cell Value

**POST** `/memories/{memory_id}/cell`

Set the value of a single cell.

**Request Body:**
```json
{
  "x": 25,
  "y": 25,
  "value": 1.0
}
```

**Response:** `200 OK`
```json
{
  "success": true,
  "x": 25,
  "y": 25,
  "value": 1.0
}
```

---

#### Batch Set Cells

**POST** `/memories/{memory_id}/batch-set`

Set multiple cell values in one request.

**Request Body:**
```json
{
  "cells": [
    {"x": 10, "y": 10, "value": 1.0},
    {"x": 11, "y": 10, "value": 0.8},
    {"x": 10, "y": 11, "value": 0.8}
  ]
}
```

**Response:** `200 OK`
```json
{
  "count": 3,
  "success": true
}
```

---

### Diffusion Operations

#### Run Diffusion

**POST** `/memories/{memory_id}/diffusion`

Execute diffusion steps on the memory.

**Request Body:**
```json
{
  "steps": 10
}
```

**Response:** `200 OK`
```json
{
  "steps_executed": 10,
  "success": true
}
```

**Description:**  
Each diffusion step spreads activation to neighbors and applies decay. Multiple steps simulate temporal evolution.

---

#### Activate Radius

**POST** `/memories/{memory_id}/activate`

Activate a circular region around a point.

**Request Body:**
```json
{
  "x": 25,
  "y": 25,
  "radius": 5,
  "strength": 1.0
}
```

**Response:** `200 OK`
```json
{
  "success": true,
  "cells_affected": 81
}
```

**Description:**  
Activates cells within radius with gradient falloff. Center gets full `strength`, edges get reduced activation.

---

### Query Operations

#### Get Neighbors

**GET** `/memories/{memory_id}/neighbors?x={x}&y={y}&radius={radius}`

Get all cells within radius of a position.

**Query Parameters:**
- `x`: Center X coordinate
- `y`: Center Y coordinate
- `radius`: Search radius

**Response:** `200 OK`
```json
{
  "neighbors": [
    {"x": 24, "y": 24, "value": 0.65},
    {"x": 25, "y": 24, "value": 0.70},
    {"x": 26, "y": 24, "value": 0.65},
    {"x": 24, "y": 25, "value": 0.70},
    {"x": 25, "y": 25, "value": 1.00},
    ...
  ]
}
```

---

#### Get Statistics

**GET** `/memories/{memory_id}/stats`

Get statistical summary of memory state.

**Response:** `200 OK`
```json
{
  "total_cells": 2500,
  "active_cells": 125,
  "active_percentage": 5.0,
  "min_value": 0.0,
  "max_value": 1.0,
  "mean_value": 0.05
}
```

**Description:**  
- `active_cells`: Cells with value > 0.01
- `active_percentage`: Percentage of active cells
- `min/max/mean_value`: Statistical values across all cells

---

### State Management

#### Get Full State

**GET** `/memories/{memory_id}/state`

Get complete memory state as 2D array.

**Response:** `200 OK`
```json
{
  "width": 50,
  "height": 50,
  "cells": [
    [0.0, 0.0, 0.1, ...],
    [0.0, 0.5, 0.8, ...],
    ...
  ],
  "diffusion_config": {
    "diffusion_rate": 0.25,
    "decay_rate": 0.1,
    "threshold": 0.01
  },
  "metadata": {}
}
```

**Note:** Can be large for big grids. Consider using specific cell queries for large memories.

---

#### Set Full State

**PUT** `/memories/{memory_id}/state`

Replace entire memory state.

**Request Body:**
```json
{
  "width": 50,
  "height": 50,
  "cells": [
    [0.0, 0.0, ...],
    ...
  ],
  "diffusion_config": {
    "diffusion_rate": 0.25,
    "decay_rate": 0.1,
    "threshold": 0.01
  },
  "metadata": {}
}
```

**Response:** `200 OK`
```json
{
  "success": true,
  "message": "Memory mem_1729625472abc state updated"
}
```

---

### Persistence

#### Save to File

**POST** `/memories/{memory_id}/save`

Save memory state to server filesystem.

**Request Body:**
```json
{
  "filename": "my_memory.json"
}
```

**Response:** `200 OK`
```json
{
  "success": true,
  "message": "Memory saved to my_memory.json"
}
```

---

#### Load from File

**POST** `/memories/{memory_id}/load`

Load memory state from server filesystem.

**Request Body:**
```json
{
  "filename": "my_memory.json"
}
```

**Response:** `200 OK`
```json
{
  "success": true,
  "message": "Memory loaded from my_memory.json"
}
```

---

## Data Models

### DiffusionConfig

```typescript
{
  diffusion_rate: number;  // How fast activation spreads (0.0-1.0)
  decay_rate: number;      // How fast activation fades (0.0-1.0)
  threshold: number;       // Minimum activation to maintain (0.0-1.0)
}
```

**Defaults:**
```json
{
  "diffusion_rate": 0.25,
  "decay_rate": 0.1,
  "threshold": 0.01
}
```

**Typical Values:**
- **Fast diffusion**: `diffusion_rate: 0.4-0.6`
- **Slow decay**: `decay_rate: 0.05-0.1`
- **Memory retention**: `threshold: 0.01-0.05`

---

### CellValue

```typescript
{
  x: number;      // X coordinate (integer)
  y: number;      // Y coordinate (integer)
  value: number;  // Cell value (typically 0.0-1.0)
}
```

---

### MemoryState

```typescript
{
  width: number;
  height: number;
  cells: number[][];  // 2D array [height][width]
  diffusion_config: DiffusionConfig;
  metadata: Record<string, string>;
}
```

---

## Examples

### Example 1: Basic Usage

```bash
# 1. Create memory
curl -X POST http://localhost:3000/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"width": 10, "height": 10}'
# Response: {"memory_id": "mem_abc123", ...}

# 2. Set initial value
curl -X POST http://localhost:3000/api/v1/memories/mem_abc123/cell \
  -H "Content-Type: application/json" \
  -d '{"x": 5, "y": 5, "value": 1.0}'

# 3. Run diffusion
curl -X POST http://localhost:3000/api/v1/memories/mem_abc123/diffusion \
  -H "Content-Type: application/json" \
  -d '{"steps": 5}'

# 4. Check neighbors
curl "http://localhost:3000/api/v1/memories/mem_abc123/neighbors?x=5&y=5&radius=1"
```

---

### Example 2: Pattern Creation

```javascript
// JavaScript/Node.js example
const BASE_URL = 'http://localhost:3000/api/v1';

// Create memory
const createResp = await fetch(`${BASE_URL}/memories`, {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({width: 50, height: 50})
});
const {memory_id} = await createResp.json();

// Create checkerboard pattern
const cells = [];
for (let y = 0; y < 50; y++) {
  for (let x = 0; x < 50; x++) {
    if ((x + y) % 2 === 0) {
      cells.push({x, y, value: 1.0});
    }
  }
}

await fetch(`${BASE_URL}/memories/${memory_id}/batch-set`, {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({cells})
});

// Run diffusion to smooth
await fetch(`${BASE_URL}/memories/${memory_id}/diffusion`, {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({steps: 10})
});

// Get final stats
const statsResp = await fetch(`${BASE_URL}/memories/${memory_id}/stats`);
const stats = await statsResp.json();
console.log(stats);
```

---

### Example 3: Python Client

```python
import requests
import json

BASE_URL = "http://localhost:3000/api/v1"

class ToroidalMemoryClient:
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url
    
    def create_memory(self, width, height):
        response = requests.post(
            f"{self.base_url}/memories",
            json={"width": width, "height": height}
        )
        return response.json()["memory_id"]
    
    def set_cell(self, memory_id, x, y, value):
        requests.post(
            f"{self.base_url}/memories/{memory_id}/cell",
            json={"x": x, "y": y, "value": value}
        )
    
    def activate_radius(self, memory_id, x, y, radius, strength):
        return requests.post(
            f"{self.base_url}/memories/{memory_id}/activate",
            json={"x": x, "y": y, "radius": radius, "strength": strength}
        ).json()
    
    def diffuse(self, memory_id, steps):
        return requests.post(
            f"{self.base_url}/memories/{memory_id}/diffusion",
            json={"steps": steps}
        ).json()
    
    def get_state(self, memory_id):
        return requests.get(
            f"{self.base_url}/memories/{memory_id}/state"
        ).json()

# Usage
client = ToroidalMemoryClient()
mem_id = client.create_memory(20, 20)
client.activate_radius(mem_id, 10, 10, 3, 1.0)
client.diffuse(mem_id, 5)
state = client.get_state(mem_id)
print(f"Active cells: {sum(sum(1 for v in row if v > 0.01) for row in state['cells'])}")
```

---

## Error Handling

All errors return appropriate HTTP status codes with JSON body:

```json
{
  "error": "Error description"
}
```

### Status Codes

| Code | Meaning | Example |
|------|---------|---------|
| 200 | Success | Request completed |
| 201 | Created | Memory instance created |
| 400 | Bad Request | Invalid parameters |
| 404 | Not Found | Memory ID doesn't exist |
| 500 | Server Error | Internal error |

### Common Errors

**Memory Not Found:**
```json
{
  "error": "Memory mem_xyz not found"
}
```

**Invalid Coordinates:**
```json
{
  "error": "Invalid coordinates"
}
```

**Serialization Error:**
```json
{
  "error": "Serialization error: ..."
}
```

---

## Rate Limiting

**Development:** No limits

**Production Recommendations:**
- 100 requests/minute per IP
- 1000 requests/hour per API key
- 10MB max request body size

---

## Performance Considerations

### Memory Size

| Size | Cells | Operations/sec | Use Case |
|------|-------|----------------|----------|
| 10×10 | 100 | >10,000 | Testing |
| 50×50 | 2,500 | >1,000 | Small apps |
| 100×100 | 10,000 | >100 | Medium apps |
| 500×500 | 250,000 | >10 | Large apps |

### Optimization Tips

1. **Batch Operations**: Use `/batch-set` instead of multiple `/cell` requests
2. **State Size**: Large grids produce large JSON responses. Use specific queries when possible.
3. **Diffusion Steps**: More steps = more computation. Typical: 5-20 steps.
4. **Active Cells**: Performance degrades with many active cells. Use decay to prune.

---

## WebSocket Support (Future)

**Coming Soon:** Real-time updates via WebSocket

```javascript
const ws = new WebSocket('ws://localhost:3000/api/v1/memories/mem_abc/watch');
ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  console.log('Memory updated:', update);
};
```

---

## Complete API Reference Table

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/memories` | Create memory instance |
| GET | `/memories` | List all memories |
| GET | `/memories/{id}` | Get memory info |
| DELETE | `/memories/{id}` | Delete memory |
| GET | `/memories/{id}/cell` | Get single cell |
| POST | `/memories/{id}/cell` | Set single cell |
| POST | `/memories/{id}/batch-set` | Set multiple cells |
| POST | `/memories/{id}/diffusion` | Run diffusion |
| POST | `/memories/{id}/activate` | Activate radius |
| GET | `/memories/{id}/neighbors` | Get neighbors |
| GET | `/memories/{id}/stats` | Get statistics |
| GET | `/memories/{id}/state` | Get full state |
| PUT | `/memories/{id}/state` | Set full state |
| POST | `/memories/{id}/save` | Save to file |
| POST | `/memories/{id}/load` | Load from file |

---

## Implementation Notes

### Running the Server

**Option 1: Demo Mode (Current)**
```bash
cargo run --example memory_server
```
Shows API logic without HTTP server.

**Option 2: Full HTTP Server**

Add to `Cargo.toml`:
```toml
[dev-dependencies]
axum = "0.7"
tokio = { version = "1", features = ["full"] }
tower = "0.4"
tower-http = { version = "0.5", features = ["cors"] }
```

Then implement Axum routes in `memory_server.rs`.

### CORS Headers

For web clients, enable CORS:
```rust
use tower_http::cors::CorsLayer;

let app = Router::new()
    .route("/memories", post(create_memory))
    .layer(CorsLayer::permissive());
```

---

## License

MIT License - Free for commercial and personal use

---

## Support

- **Documentation**: See project README and examples
- **Issues**: GitHub Issues
- **Examples**: `/examples/memory_server.rs`

---

*API Version 1.0 - Last Updated: October 22, 2025*
