/// REST API Server for Toroidal Memory
/// 
/// A HTTP/JSON API for managing toroidal memory instances with persistence.
/// 
/// Features:
/// - Create/delete memory instances
/// - Get/set individual cells
/// - Batch operations
/// - Diffusion simulation
/// - Pattern matching
/// - Save/load state
/// - Query neighbors
/// 
/// Run with: cargo run --example memory_server
/// 
/// Dependencies (add to Cargo.toml for this example):
/// ```toml
/// [dev-dependencies]
/// axum = "0.7"
/// tokio = { version = "1", features = ["full"] }
/// tower = "0.4"
/// tower-http = { version = "0.5", features = ["cors"] }
/// ```

use ai_toroidal_memory::{ToroidalMemory, DiffusionEngine, DiffusionConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::fs;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post, put, delete},
    Json, Router,
};
use tower_http::cors::CorsLayer;
use std::net::SocketAddr;

// ============================================================================
// Data Structures
// ============================================================================

/// Memory instance state
#[derive(Clone)]
struct MemoryInstance {
    memory: ToroidalMemory<f64>,
    diffusion_config: DiffusionConfig,
    metadata: HashMap<String, String>,
}

/// Global memory store
type MemoryStore = Arc<RwLock<HashMap<String, MemoryInstance>>>;

/// Serializable memory state
#[derive(Serialize, Deserialize)]
struct MemoryState {
    width: usize,
    height: usize,
    cells: Vec<Vec<f64>>,
    diffusion_config: DiffusionConfigDto,
    metadata: HashMap<String, String>,
}

#[derive(Serialize, Deserialize, Clone)]
struct DiffusionConfigDto {
    diffusion_rate: f64,
    decay_rate: f64,
    threshold: f64,
}

impl From<DiffusionConfig> for DiffusionConfigDto {
    fn from(config: DiffusionConfig) -> Self {
        DiffusionConfigDto {
            diffusion_rate: config.diffusion_rate,
            decay_rate: config.decay_rate,
            threshold: config.threshold,
        }
    }
}

impl From<DiffusionConfigDto> for DiffusionConfig {
    fn from(dto: DiffusionConfigDto) -> Self {
        DiffusionConfig {
            diffusion_rate: dto.diffusion_rate,
            decay_rate: dto.decay_rate,
            threshold: dto.threshold,
        }
    }
}

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Deserialize)]
struct CreateMemoryRequest {
    width: usize,
    height: usize,
    #[serde(default)]
    diffusion_config: Option<DiffusionConfigDto>,
    #[serde(default)]
    metadata: HashMap<String, String>,
}

#[derive(Serialize)]
struct CreateMemoryResponse {
    memory_id: String,
    width: usize,
    height: usize,
    message: String,
}

#[derive(Deserialize)]
struct SetCellRequest {
    x: isize,
    y: isize,
    value: f64,
}

#[derive(Serialize)]
struct SetCellResponse {
    success: bool,
    x: isize,
    y: isize,
    value: f64,
}

#[derive(Deserialize)]
struct GetCellRequest {
    x: isize,
    y: isize,
}

#[derive(Serialize)]
struct GetCellResponse {
    x: isize,
    y: isize,
    value: f64,
}

#[derive(Deserialize)]
struct BatchSetRequest {
    cells: Vec<CellValue>,
}

#[derive(Deserialize, Serialize)]
struct CellValue {
    x: isize,
    y: isize,
    value: f64,
}

#[derive(Serialize)]
struct BatchSetResponse {
    count: usize,
    success: bool,
}

#[derive(Deserialize)]
struct DiffusionRequest {
    steps: usize,
}

#[derive(Serialize)]
struct DiffusionResponse {
    steps_executed: usize,
    success: bool,
}

#[derive(Deserialize)]
struct ActivateRadiusRequest {
    x: isize,
    y: isize,
    radius: isize,
    strength: f64,
}

#[derive(Serialize)]
struct ActivateRadiusResponse {
    success: bool,
    cells_affected: usize,
}

#[derive(Deserialize)]
struct GetNeighborsRequest {
    x: isize,
    y: isize,
    radius: isize,
}

#[derive(Serialize)]
struct GetNeighborsResponse {
    neighbors: Vec<CellValue>,
}

#[derive(Serialize)]
struct MemoryInfoResponse {
    memory_id: String,
    width: usize,
    height: usize,
    diffusion_config: DiffusionConfigDto,
    metadata: HashMap<String, String>,
}

#[derive(Serialize)]
struct ListMemoriesResponse {
    memories: Vec<String>,
    count: usize,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Serialize)]
struct SuccessResponse {
    success: bool,
    message: String,
}

#[derive(Serialize)]
struct StatsResponse {
    total_cells: usize,
    active_cells: usize,
    active_percentage: f64,
    min_value: f64,
    max_value: f64,
    mean_value: f64,
}

// ============================================================================
// API Implementation (without web framework dependencies)
// ============================================================================

struct MemoryAPI {
    store: MemoryStore,
}

impl MemoryAPI {
    fn new() -> Self {
        MemoryAPI {
            store: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    // POST /memories
    fn create_memory(&self, req: CreateMemoryRequest) -> Result<CreateMemoryResponse, String> {
        let memory_id = format!("mem_{}", uuid());
        
        let diffusion_config = req.diffusion_config
            .map(|c| c.into())
            .unwrap_or(DiffusionConfig {
                diffusion_rate: 0.25,
                decay_rate: 0.1,
                threshold: 0.01,
            });
        
        let instance = MemoryInstance {
            memory: ToroidalMemory::new(req.width, req.height),
            diffusion_config,
            metadata: req.metadata,
        };
        
        let mut store = self.store.write().unwrap();
        store.insert(memory_id.clone(), instance);
        
        Ok(CreateMemoryResponse {
            memory_id: memory_id.clone(),
            width: req.width,
            height: req.height,
            message: format!("Memory {} created successfully", memory_id),
        })
    }
    
    // GET /memories
    fn list_memories(&self) -> ListMemoriesResponse {
        let store = self.store.read().unwrap();
        let memories: Vec<String> = store.keys().cloned().collect();
        let count = memories.len();
        
        ListMemoriesResponse { memories, count }
    }
    
    // GET /memories/{id}
    fn get_memory_info(&self, memory_id: &str) -> Result<MemoryInfoResponse, String> {
        let store = self.store.read().unwrap();
        let instance = store.get(memory_id)
            .ok_or_else(|| format!("Memory {} not found", memory_id))?;
        
        let (width, height) = instance.memory.dimensions();
        
        Ok(MemoryInfoResponse {
            memory_id: memory_id.to_string(),
            width,
            height,
            diffusion_config: instance.diffusion_config.into(),
            metadata: instance.metadata.clone(),
        })
    }
    
    // DELETE /memories/{id}
    fn delete_memory(&self, memory_id: &str) -> Result<SuccessResponse, String> {
        let mut store = self.store.write().unwrap();
        store.remove(memory_id)
            .ok_or_else(|| format!("Memory {} not found", memory_id))?;
        
        Ok(SuccessResponse {
            success: true,
            message: format!("Memory {} deleted", memory_id),
        })
    }
    
    // GET /memories/{id}/cell
    fn get_cell(&self, memory_id: &str, req: GetCellRequest) -> Result<GetCellResponse, String> {
        let store = self.store.read().unwrap();
        let instance = store.get(memory_id)
            .ok_or_else(|| format!("Memory {} not found", memory_id))?;
        
        let value = *instance.memory.get(req.x, req.y)
            .ok_or_else(|| "Invalid coordinates".to_string())?;
        
        Ok(GetCellResponse {
            x: req.x,
            y: req.y,
            value,
        })
    }
    
    // POST /memories/{id}/cell
    fn set_cell(&self, memory_id: &str, req: SetCellRequest) -> Result<SetCellResponse, String> {
        let mut store = self.store.write().unwrap();
        let instance = store.get_mut(memory_id)
            .ok_or_else(|| format!("Memory {} not found", memory_id))?;
        
        instance.memory.set(req.x, req.y, req.value);
        
        Ok(SetCellResponse {
            success: true,
            x: req.x,
            y: req.y,
            value: req.value,
        })
    }
    
    // POST /memories/{id}/batch-set
    fn batch_set(&self, memory_id: &str, req: BatchSetRequest) -> Result<BatchSetResponse, String> {
        let mut store = self.store.write().unwrap();
        let instance = store.get_mut(memory_id)
            .ok_or_else(|| format!("Memory {} not found", memory_id))?;
        
        for cell in &req.cells {
            instance.memory.set(cell.x, cell.y, cell.value);
        }
        
        Ok(BatchSetResponse {
            count: req.cells.len(),
            success: true,
        })
    }
    
    // POST /memories/{id}/diffusion
    fn run_diffusion(&self, memory_id: &str, req: DiffusionRequest) -> Result<DiffusionResponse, String> {
        let mut store = self.store.write().unwrap();
        let instance = store.get_mut(memory_id)
            .ok_or_else(|| format!("Memory {} not found", memory_id))?;
        
        let engine = DiffusionEngine::new(instance.diffusion_config);
        
        for _ in 0..req.steps {
            engine.step(&mut instance.memory);
        }
        
        Ok(DiffusionResponse {
            steps_executed: req.steps,
            success: true,
        })
    }
    
    // POST /memories/{id}/activate
    fn activate_radius(&self, memory_id: &str, req: ActivateRadiusRequest) -> Result<ActivateRadiusResponse, String> {
        let mut store = self.store.write().unwrap();
        let instance = store.get_mut(memory_id)
            .ok_or_else(|| format!("Memory {} not found", memory_id))?;
        
        let mut cells_affected = 0;
        
        for dy in -req.radius..=req.radius {
            for dx in -req.radius..=req.radius {
                let dist = ((dx * dx + dy * dy) as f64).sqrt();
                if dist <= req.radius as f64 {
                    let activation_strength = req.strength * (1.0 - dist / (req.radius as f64 + 1.0));
                    let current = *instance.memory.get(req.x + dx, req.y + dy).unwrap();
                    instance.memory.set(req.x + dx, req.y + dy, current.max(activation_strength));
                    cells_affected += 1;
                }
            }
        }
        
        Ok(ActivateRadiusResponse {
            success: true,
            cells_affected,
        })
    }
    
    // GET /memories/{id}/neighbors
    fn get_neighbors(&self, memory_id: &str, req: GetNeighborsRequest) -> Result<GetNeighborsResponse, String> {
        let store = self.store.read().unwrap();
        let instance = store.get(memory_id)
            .ok_or_else(|| format!("Memory {} not found", memory_id))?;
        
        let neighbors_data = instance.memory.get_neighbors(req.x, req.y, req.radius);
        
        let neighbors: Vec<CellValue> = neighbors_data.iter()
            .map(|(x, y, value)| CellValue {
                x: *x,
                y: *y,
                value: **value,
            })
            .collect();
        
        Ok(GetNeighborsResponse { neighbors })
    }
    
    // GET /memories/{id}/state
    fn get_state(&self, memory_id: &str) -> Result<MemoryState, String> {
        let store = self.store.read().unwrap();
        let instance = store.get(memory_id)
            .ok_or_else(|| format!("Memory {} not found", memory_id))?;
        
        let (width, height) = instance.memory.dimensions();
        let mut cells = vec![vec![0.0; width]; height];
        
        for y in 0..height {
            for x in 0..width {
                cells[y][x] = *instance.memory.get(x as isize, y as isize).unwrap();
            }
        }
        
        Ok(MemoryState {
            width,
            height,
            cells,
            diffusion_config: instance.diffusion_config.into(),
            metadata: instance.metadata.clone(),
        })
    }
    
    // PUT /memories/{id}/state
    fn set_state(&self, memory_id: &str, state: MemoryState) -> Result<SuccessResponse, String> {
        let mut store = self.store.write().unwrap();
        
        let mut memory = ToroidalMemory::new(state.width, state.height);
        
        for (y, row) in state.cells.iter().enumerate() {
            for (x, &value) in row.iter().enumerate() {
                memory.set(x as isize, y as isize, value);
            }
        }
        
        let instance = MemoryInstance {
            memory,
            diffusion_config: state.diffusion_config.into(),
            metadata: state.metadata,
        };
        
        store.insert(memory_id.to_string(), instance);
        
        Ok(SuccessResponse {
            success: true,
            message: format!("Memory {} state updated", memory_id),
        })
    }
    
    // GET /memories/{id}/stats
    fn get_stats(&self, memory_id: &str) -> Result<StatsResponse, String> {
        let store = self.store.read().unwrap();
        let instance = store.get(memory_id)
            .ok_or_else(|| format!("Memory {} not found", memory_id))?;
        
        let (width, height) = instance.memory.dimensions();
        let total_cells = width * height;
        
        let mut active_cells = 0;
        let mut min_value = f64::INFINITY;
        let mut max_value = f64::NEG_INFINITY;
        let mut sum = 0.0;
        
        for y in 0..height {
            for x in 0..width {
                let value = *instance.memory.get(x as isize, y as isize).unwrap();
                if value > 0.01 {
                    active_cells += 1;
                }
                min_value = min_value.min(value);
                max_value = max_value.max(value);
                sum += value;
            }
        }
        
        let mean_value = sum / total_cells as f64;
        let active_percentage = (active_cells as f64 / total_cells as f64) * 100.0;
        
        Ok(StatsResponse {
            total_cells,
            active_cells,
            active_percentage,
            min_value,
            max_value,
            mean_value,
        })
    }
    
    // POST /memories/{id}/save
    fn save_to_file(&self, memory_id: &str, filename: &str) -> Result<SuccessResponse, String> {
        let state = self.get_state(memory_id)?;
        let json = serde_json::to_string_pretty(&state)
            .map_err(|e| format!("Serialization error: {}", e))?;
        
        fs::write(filename, json)
            .map_err(|e| format!("File write error: {}", e))?;
        
        Ok(SuccessResponse {
            success: true,
            message: format!("Memory saved to {}", filename),
        })
    }
    
    // POST /memories/{id}/load
    fn load_from_file(&self, memory_id: &str, filename: &str) -> Result<SuccessResponse, String> {
        let json = fs::read_to_string(filename)
            .map_err(|e| format!("File read error: {}", e))?;
        
        let state: MemoryState = serde_json::from_str(&json)
            .map_err(|e| format!("Deserialization error: {}", e))?;
        
        self.set_state(memory_id, state)?;
        
        Ok(SuccessResponse {
            success: true,
            message: format!("Memory loaded from {}", filename),
        })
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn uuid() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    format!("{:x}", timestamp)
}

// ============================================================================
// HTTP Server Implementation
// ============================================================================

type AppState = Arc<MemoryAPI>;

// Error response wrapper
struct AppError(String);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: self.0 })).into_response()
    }
}

impl<E: std::fmt::Display> From<E> for AppError {
    fn from(err: E) -> Self {
        AppError(err.to_string())
    }
}

// HTTP Handlers

async fn create_memory_handler(
    State(api): State<AppState>,
    Json(req): Json<CreateMemoryRequest>,
) -> Result<Json<CreateMemoryResponse>, AppError> {
    let resp = api.create_memory(req)?;
    Ok(Json(resp))
}

async fn list_memories_handler(
    State(api): State<AppState>,
) -> Result<Json<ListMemoriesResponse>, AppError> {
    Ok(Json(api.list_memories()))
}

async fn get_memory_info_handler(
    State(api): State<AppState>,
    Path(memory_id): Path<String>,
) -> Result<Json<MemoryInfoResponse>, AppError> {
    let resp = api.get_memory_info(&memory_id)?;
    Ok(Json(resp))
}

async fn delete_memory_handler(
    State(api): State<AppState>,
    Path(memory_id): Path<String>,
) -> Result<Json<SuccessResponse>, AppError> {
    let resp = api.delete_memory(&memory_id)?;
    Ok(Json(resp))
}

#[derive(Deserialize)]
struct CellQuery {
    x: isize,
    y: isize,
}

async fn get_cell_handler(
    State(api): State<AppState>,
    Path(memory_id): Path<String>,
    Query(query): Query<CellQuery>,
) -> Result<Json<CellValue>, AppError> {
    let store = api.store.read().unwrap();
    let instance = store.get(&memory_id)
        .ok_or_else(|| AppError(format!("Memory {} not found", memory_id)))?;
    
    let value = *instance.memory.get(query.x, query.y)
        .ok_or_else(|| AppError("Invalid coordinates".to_string()))?;
    
    Ok(Json(CellValue {
        x: query.x,
        y: query.y,
        value,
    }))
}

async fn set_cell_handler(
    State(api): State<AppState>,
    Path(memory_id): Path<String>,
    Json(req): Json<SetCellRequest>,
) -> Result<Json<SetCellResponse>, AppError> {
    let resp = api.set_cell(&memory_id, req)?;
    Ok(Json(resp))
}

async fn batch_set_handler(
    State(api): State<AppState>,
    Path(memory_id): Path<String>,
    Json(req): Json<BatchSetRequest>,
) -> Result<Json<BatchSetResponse>, AppError> {
    let resp = api.batch_set(&memory_id, req)?;
    Ok(Json(resp))
}

async fn run_diffusion_handler(
    State(api): State<AppState>,
    Path(memory_id): Path<String>,
    Json(req): Json<DiffusionRequest>,
) -> Result<Json<DiffusionResponse>, AppError> {
    let resp = api.run_diffusion(&memory_id, req)?;
    Ok(Json(resp))
}

async fn activate_radius_handler(
    State(api): State<AppState>,
    Path(memory_id): Path<String>,
    Json(req): Json<ActivateRadiusRequest>,
) -> Result<Json<ActivateRadiusResponse>, AppError> {
    let resp = api.activate_radius(&memory_id, req)?;
    Ok(Json(resp))
}

#[derive(Deserialize)]
struct NeighborsQuery {
    x: isize,
    y: isize,
    radius: isize,
}

async fn get_neighbors_handler(
    State(api): State<AppState>,
    Path(memory_id): Path<String>,
    Query(query): Query<NeighborsQuery>,
) -> Result<Json<GetNeighborsResponse>, AppError> {
    let req = GetNeighborsRequest {
        x: query.x,
        y: query.y,
        radius: query.radius,
    };
    let resp = api.get_neighbors(&memory_id, req)?;
    Ok(Json(resp))
}

async fn get_stats_handler(
    State(api): State<AppState>,
    Path(memory_id): Path<String>,
) -> Result<Json<StatsResponse>, AppError> {
    let resp = api.get_stats(&memory_id)?;
    Ok(Json(resp))
}

async fn get_state_handler(
    State(api): State<AppState>,
    Path(memory_id): Path<String>,
) -> Result<Json<MemoryState>, AppError> {
    let resp = api.get_state(&memory_id)?;
    Ok(Json(resp))
}

async fn set_state_handler(
    State(api): State<AppState>,
    Path(memory_id): Path<String>,
    Json(state): Json<MemoryState>,
) -> Result<Json<SuccessResponse>, AppError> {
    let resp = api.set_state(&memory_id, state)?;
    Ok(Json(resp))
}

#[derive(Deserialize)]
struct FileRequest {
    filename: String,
}

async fn save_file_handler(
    State(api): State<AppState>,
    Path(memory_id): Path<String>,
    Json(req): Json<FileRequest>,
) -> Result<Json<SuccessResponse>, AppError> {
    let resp = api.save_to_file(&memory_id, &req.filename)?;
    Ok(Json(resp))
}

async fn load_file_handler(
    State(api): State<AppState>,
    Path(memory_id): Path<String>,
    Json(req): Json<FileRequest>,
) -> Result<Json<SuccessResponse>, AppError> {
    let resp = api.load_from_file(&memory_id, &req.filename)?;
    Ok(Json(resp))
}

async fn health_check() -> &'static str {
    "OK"
}

#[tokio::main]
async fn main() {
    // Get storage path from environment or use default
    let storage_path = std::env::var("STORAGE_PATH").unwrap_or_else(|_| "./data".to_string());
    
    // Create storage directory if it doesn't exist
    std::fs::create_dir_all(&storage_path)
        .expect("Failed to create storage directory");
    
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║       Toroidal Memory REST API Server                     ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    println!("📁 Storage path: {}", storage_path);
    println!("🚀 Starting server...");
    println!();
    
    // Change to storage directory for file operations
    std::env::set_current_dir(&storage_path)
        .expect("Failed to change to storage directory");
    
    let api = Arc::new(MemoryAPI::new());
    
    // Build router
    let app = Router::new()
        // Health check
        .route("/health", get(health_check))
        
        // Memory management
        .route("/api/v1/memories", post(create_memory_handler))
        .route("/api/v1/memories", get(list_memories_handler))
        .route("/api/v1/memories/:id", get(get_memory_info_handler))
        .route("/api/v1/memories/:id", delete(delete_memory_handler))
        
        // Cell operations
        .route("/api/v1/memories/:id/cell", get(get_cell_handler))
        .route("/api/v1/memories/:id/cell", post(set_cell_handler))
        .route("/api/v1/memories/:id/batch-set", post(batch_set_handler))
        
        // Diffusion operations
        .route("/api/v1/memories/:id/diffusion", post(run_diffusion_handler))
        .route("/api/v1/memories/:id/activate", post(activate_radius_handler))
        
        // Query operations
        .route("/api/v1/memories/:id/neighbors", get(get_neighbors_handler))
        .route("/api/v1/memories/:id/stats", get(get_stats_handler))
        
        // State management
        .route("/api/v1/memories/:id/state", get(get_state_handler))
        .route("/api/v1/memories/:id/state", put(set_state_handler))
        
        // File operations
        .route("/api/v1/memories/:id/save", post(save_file_handler))
        .route("/api/v1/memories/:id/load", post(load_file_handler))
        
        // Add CORS
        .layer(CorsLayer::permissive())
        
        // Add shared state
        .with_state(api);
    
    // Bind to address
    let port = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(3000);
    
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    
    println!("✅ Server running on http://{}", addr);
    println!("📖 API documentation: docs/API_DOCUMENTATION.md");
    println!();
    println!("Example requests:");
    println!("  curl http://localhost:{}/health", port);
    println!("  curl -X POST http://localhost:{}/api/v1/memories \\", port);
    println!("    -H 'Content-Type: application/json' \\");
    println!("    -d '{{\"width\": 50, \"height\": 50}}'");
    println!();
    
    // Start server
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind to address");
    
    axum::serve(listener, app)
        .await
        .expect("Server failed");
}
