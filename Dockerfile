# =============================================================================
# Dockerfile for CIM Grid Control Engine
# Multi-stage build for optimized production deployment
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies and compile if needed
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

# Set build-time environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies for numerical libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy only requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --user --no-warn-script-location -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Minimal production image
# -----------------------------------------------------------------------------
FROM python:3.11-slim

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/root/.local/bin:$PATH \
    MPLBACKEND=Agg

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 gridengine && \
    mkdir -p /app && \
    chown -R gridengine:gridengine /app

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY --chown=gridengine:gridengine . .

# Switch to non-root user
USER gridengine

# Health check (optional - useful for orchestration)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import numpy, networkx, pydantic, scipy; print('OK')" || exit 1

# Default command - run the grid simulation
CMD ["python", "main.py"]

# Metadata labels (following OCI standards)
LABEL org.opencontainers.image.title="CIM Grid Control Engine" \
      org.opencontainers.image.description="Hybrid physics-software engine for distribution grid analysis" \
      org.opencontainers.image.vendor="omari91" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/omari91/cim-grid-control-engine"
