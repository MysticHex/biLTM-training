"""
FastAPI Backend for AttnRetrofit Dashboard

Endpoints:
- GET /api/health - Health check
- GET /api/metrics - Model performance metrics
- GET /api/buildings - Building list with anomaly scores
- GET /api/building/{id} - Single building details
- GET /api/predictions/{id} - Predictions for a building
- GET /api/attention/{id} - Attention weights for visualization
- POST /api/predict - Make prediction for new data

Run:
    uvicorn api_server:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
from typing import List, Optional
import sys

app = FastAPI(
    title="AttnRetrofit API",
    description="API for Smart Building Retrofit Prioritization Dashboard",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded data
DATA = {
    "model": None,
    "config": None,
    "test_data": None,
    "building_meta": None,
    "anomaly_report": None,
    "metrics": None,
    "predictions": None,
    "attention_weights": None
}

# Data directory
DATA_DIR = Path(__file__).parent


def load_data():
    """Load all necessary data on startup"""
    print("🔄 Loading data...")
    
    try:
        # Load test metrics
        metrics_path = DATA_DIR / "test_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                DATA["metrics"] = json.load(f)
            print("✅ Loaded test_metrics.json")
        
        # Load anomaly report
        anomaly_path = DATA_DIR / "anomaly_report.csv"
        if anomaly_path.exists():
            DATA["anomaly_report"] = pd.read_csv(anomaly_path)
            print("✅ Loaded anomaly_report.csv")
        
        # Load retrofit priority report
        retrofit_path = DATA_DIR / "retrofit_priority_report.csv"
        if retrofit_path.exists():
            DATA["retrofit_report"] = pd.read_csv(retrofit_path)
            print("✅ Loaded retrofit_priority_report.csv")
        
        # Load building metadata
        meta_path = DATA_DIR / "data" / "building_metadata.csv"
        if meta_path.exists():
            DATA["building_meta"] = pd.read_csv(meta_path)
            print("✅ Loaded building_metadata.csv")
        
        # Load test predictions
        test_seq_path = DATA_DIR / "processed_test_seq.npy"
        test_tgt_path = DATA_DIR / "processed_test_tgt.npy"
        test_bid_path = DATA_DIR / "processed_test_bid.npy"
        
        if all(p.exists() for p in [test_seq_path, test_tgt_path, test_bid_path]):
            DATA["test_data"] = {
                "sequences": np.load(test_seq_path),
                "targets": np.load(test_tgt_path),
                "building_ids": np.load(test_bid_path)
            }
            print("✅ Loaded test data")
        
        # Load best params
        params_path = DATA_DIR / "best_params.json"
        if params_path.exists():
            with open(params_path, 'r') as f:
                DATA["config"] = json.load(f)
            print("✅ Loaded best_params.json")
        
        print("✅ All data loaded successfully!")
        
    except Exception as e:
        print(f"⚠️ Error loading data: {e}")


@app.on_event("startup")
async def startup_event():
    load_data()


# ============== API ENDPOINTS ==============

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": DATA["config"] is not None,
        "data_loaded": DATA["anomaly_report"] is not None
    }


@app.get("/api/metrics")
async def get_metrics():
    """Get model performance metrics"""
    if DATA["metrics"] is None:
        return {
            "RMSE": 0,
            "MAE": 0,
            "RMSLE": 0,
            "message": "Metrics not loaded"
        }
    return DATA["metrics"]


@app.get("/api/config")
async def get_config():
    """Get model configuration"""
    if DATA["config"] is None:
        return {"message": "Config not loaded"}
    return DATA["config"]


@app.get("/api/buildings")
async def get_buildings(
    limit: int = 50,
    sort_by: str = "combined_score",
    order: str = "desc"
):
    """Get list of buildings with anomaly scores"""
    if DATA["anomaly_report"] is None:
        raise HTTPException(status_code=404, detail="Anomaly report not found")
    
    df = DATA["anomaly_report"].copy()
    
    # Merge with building metadata if available
    if DATA["building_meta"] is not None:
        df = df.merge(
            DATA["building_meta"][['building_id', 'primary_use', 'square_feet', 'year_built', 'site_id']],
            on='building_id',
            how='left'
        )
    
    # Sort
    ascending = order.lower() == "asc"
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)
    
    # Limit
    df = df.head(limit)
    
    # Convert to list of dicts
    buildings = df.to_dict(orient='records')
    
    # Clean NaN values
    for b in buildings:
        for k, v in b.items():
            if pd.isna(v):
                b[k] = None
    
    return {
        "total": len(DATA["anomaly_report"]),
        "returned": len(buildings),
        "buildings": buildings
    }


@app.get("/api/building/{building_id}")
async def get_building(building_id: int):
    """Get details for a single building"""
    if DATA["anomaly_report"] is None:
        raise HTTPException(status_code=404, detail="Data not found")
    
    # Find in anomaly report
    df = DATA["anomaly_report"]
    building_data = df[df['building_id'] == building_id]
    
    if len(building_data) == 0:
        raise HTTPException(status_code=404, detail=f"Building {building_id} not found")
    
    result = building_data.iloc[0].to_dict()
    
    # Add metadata
    if DATA["building_meta"] is not None:
        meta = DATA["building_meta"]
        meta_row = meta[meta['building_id'] == building_id]
        if len(meta_row) > 0:
            for col in ['primary_use', 'square_feet', 'year_built', 'site_id', 'floor_count']:
                if col in meta_row.columns:
                    result[col] = meta_row.iloc[0][col]
    
    # Clean NaN
    for k, v in result.items():
        if pd.isna(v):
            result[k] = None
    
    return result


@app.get("/api/predictions/{building_id}")
async def get_predictions(building_id: int):
    """Get predictions for a building"""
    if DATA["test_data"] is None:
        raise HTTPException(status_code=404, detail="Test data not loaded")
    
    # Find indices for this building
    building_ids = DATA["test_data"]["building_ids"]
    indices = np.where(building_ids == building_id)[0]
    
    if len(indices) == 0:
        raise HTTPException(status_code=404, detail=f"No predictions for building {building_id}")
    
    # Get last sequence for this building
    idx = indices[-1]
    
    return {
        "building_id": int(building_id),
        "target": DATA["test_data"]["targets"][idx].tolist(),
        "sequence_length": int(DATA["test_data"]["sequences"].shape[1]),
        "prediction_horizon": int(DATA["test_data"]["targets"].shape[1])
    }


@app.get("/api/summary")
async def get_summary():
    """Get dashboard summary statistics"""
    summary = {
        "total_buildings": 0,
        "high_priority": 0,
        "medium_priority": 0,
        "low_priority": 0,
        "avg_anomaly_score": 0,
        "total_potential_savings": 0,
        "building_types": {}
    }
    
    if DATA["anomaly_report"] is not None:
        df = DATA["anomaly_report"]
        summary["total_buildings"] = len(df)
        summary["avg_anomaly_score"] = float(df['combined_score'].mean())
        
        # Priority counts (based on percentiles)
        p95 = df['combined_score'].quantile(0.95)
        p75 = df['combined_score'].quantile(0.75)
        summary["high_priority"] = int((df['combined_score'] > p95).sum())
        summary["medium_priority"] = int(((df['combined_score'] > p75) & (df['combined_score'] <= p95)).sum())
        summary["low_priority"] = int((df['combined_score'] <= p75).sum())
    
    if DATA.get("retrofit_report") is not None:
        df = DATA["retrofit_report"]
        if 'potential_savings_kwh' in df.columns:
            summary["total_potential_savings"] = float(df['potential_savings_kwh'].sum())
    
    if DATA["building_meta"] is not None:
        type_counts = DATA["building_meta"]['primary_use'].value_counts().head(10).to_dict()
        summary["building_types"] = type_counts
    
    return summary


@app.get("/api/chart/anomaly-distribution")
async def get_anomaly_distribution():
    """Get anomaly score distribution for charts"""
    if DATA["anomaly_report"] is None:
        return {"bins": [], "counts": []}
    
    scores = DATA["anomaly_report"]['combined_score'].values
    hist, bin_edges = np.histogram(scores, bins=20)
    
    return {
        "bins": bin_edges.tolist(),
        "counts": hist.tolist(),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max())
    }


@app.get("/api/chart/by-building-type")
async def get_by_building_type():
    """Get anomaly stats by building type"""
    if DATA["anomaly_report"] is None or DATA["building_meta"] is None:
        return {"types": [], "scores": [], "counts": []}
    
    df = DATA["anomaly_report"].merge(
        DATA["building_meta"][['building_id', 'primary_use']],
        on='building_id',
        how='left'
    )
    
    grouped = df.groupby('primary_use').agg({
        'combined_score': 'mean',
        'building_id': 'count'
    }).reset_index()
    
    grouped = grouped.sort_values('combined_score', ascending=False)
    
    return {
        "types": grouped['primary_use'].fillna('Unknown').tolist(),
        "scores": grouped['combined_score'].tolist(),
        "counts": grouped['building_id'].tolist()
    }


@app.get("/api/chart/top-anomalies")
async def get_top_anomalies(limit: int = 10):
    """Get top anomalous buildings"""
    if DATA["anomaly_report"] is None:
        return {"buildings": [], "scores": []}
    
    df = DATA["anomaly_report"].nlargest(limit, 'combined_score')
    
    return {
        "buildings": df['building_id'].astype(str).tolist(),
        "scores": df['combined_score'].tolist(),
        "residuals": df['residual_mean'].tolist() if 'residual_mean' in df.columns else []
    }


# Serve dashboard.html at root
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the dashboard HTML"""
    dashboard_path = DATA_DIR / "dashboard.html"
    if dashboard_path.exists():
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            return f.read()
    return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)


if __name__ == "__main__":
    import uvicorn
    print("""
================================================================================
🚀 AttnRetrofit Dashboard Server
================================================================================

Starting server at: http://localhost:8000

Open your browser and go to:
  → http://localhost:8000        (Dashboard)
  → http://localhost:8000/docs   (API Documentation)

Press Ctrl+C to stop the server.
================================================================================
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000)
