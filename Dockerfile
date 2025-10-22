# Multi-stage build for Rust application
FROM rust:1.90 as builder

WORKDIR /usr/src/app

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src ./src
COPY examples ./examples

# Build the application in release mode
RUN cargo build --release --example memory_server

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 appuser

# Create data directory
RUN mkdir -p /data && chown appuser:appuser /data

# Copy the binary from builder
COPY --from=builder /usr/src/app/target/release/examples/memory_server /usr/local/bin/memory_server

# Set user
USER appuser

# Set working directory
WORKDIR /data

# Expose port
EXPOSE 3000

# Set environment variables
ENV STORAGE_PATH=/data
ENV PORT=3000
ENV RUST_LOG=info

# Run the server
CMD ["memory_server"]
