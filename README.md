# Distribution Grid Optimization Engine

## Executive Summary
This project implements a hybrid physics-software engine for analyzing low-voltage distribution grids. It was engineered to address specific DSO challenges such as reverse power flows and voltage violations caused by Distributed Energy Resources (DERs).

## Key Features
- **Physics Core:** Forward-Backward Sweep (FBS) solver optimized for high R/X ratio radial feeders.
- **Domain Modeling:** IEC 61970 (CIM) compliant data models using Pydantic for robust type validation.
- **Advanced Controls:** Fuzzy Logic Controller for Volt-VAR optimization in PV inverters.
- **Analytics:** Hosting Capacity Analysis using binary search ($O(\log N)$) for rapid connection assessment.

## Architecture
- **Layer 1: Domain Data (Pydantic):** Enforces physical constraints (e.g., non-negative resistance) at the I/O boundary.
- **Layer 2: Topology (NetworkX):** Validates radiality and determines feeder ordering.
- **Layer 3: Solvers (SciPy/NumPy):** Implements numerical methods for non-linear power flow.
- **Layer 4: Application:** Orchestrates solvers to answer business questions (e.g., "Can I connect this EV charger?").

## Engineering Assumptions & Future Roadmap

### Physical Assumptions
- **Three-Phase Balance:** The solver currently assumes a perfectly balanced three-phase system and models the positive sequence network.
- **Topology:** The Forward-Backward Sweep (FBS) implementation is optimized for **radial** distribution feeders (trees). Meshed topologies would require a Newton-Raphson solver upgrade.

### Production Considerations (TODOs)
- **Scaling:** The current BFS parent lookup is $O(N^2)$ in the worst case. For production grids (>10k nodes), this would be optimized to $O(N)$ by caching the tree structure.
- **Concurrency:** The hosting capacity algorithm currently mutates the grid state in-place. A production version would use immutable data structures to ensure thread safety in a REST API context.
- **Grid Codes:** The Fuzzy Logic Controller is a conceptual demonstration. Real-world implementation would require adherence to VDE-AR-N 4105 static voltage support curves.

## Quick Start

### Option 1: Using Python Directly (Development)
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

   Run the simulation and validation suite:

Bash

python main.py
## rem to update 0. CONFIGURATION & CONSTANTS before runing

### Option 2: Using Docker (Recommended for Production)

Docker provides an isolated, reproducible environment for running the grid engine:

**Build and run with docker-compose (easiest):**
```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

**Or using Docker directly:**
```bash
# Build the image
docker build -t cim-grid-engine .

# Run a simulation
docker run --rm cim-grid-engine

# Run with output volume mounted
mkdir -p output
docker run --rm -v $(pwd)/output:/app/output cim-grid-engine
```

**Benefits of Docker deployment:**
- ✅ Consistent environment across systems (Windows/Mac/Linux)
- ✅ No dependency conflicts
- ✅ Production-ready containerization
- ✅ Easy integration with CI/CD pipelines
- ✅ Scalable for distributed grid analysis
