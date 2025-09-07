"""FastAPI application for threat detection serving."""

from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger

from ..common.config import get_config
from .inference import create_inference_engine


# Pydantic models
class ThreatSample(BaseModel):
    """Input threat sample model."""
    Time: Optional[str] = None
    Protocol: Optional[str] = None
    Flag: Optional[str] = None
    Family: Optional[str] = None
    SourceAddress: Optional[str] = None
    DestinationAddress: Optional[str] = None
    Port: Optional[int] = None
    PayloadSize: Optional[int] = Field(None, alias="Payload Size")
    NumberOfPackets: Optional[int] = Field(None, alias="Number of Packets")
    ApplicationLayerData: Optional[str] = Field(None, alias="Application Layer Data")
    UserAgent: Optional[str] = Field(None, alias="User-Agent")
    Geolocation: Optional[str] = None
    AnomalyScore: Optional[float] = Field(None, alias="Anomaly Score")
    EventDescription: Optional[str] = Field(None, alias="Event Description")
    ResponseTime: Optional[float] = Field(None, alias="Response Time")
    DataTransferRate: Optional[float] = Field(None, alias="Data Transfer Rate")
    
    class Config:
        allow_population_by_field_name = True


class ThreatPrediction(BaseModel):
    """Output prediction model."""
    prediction: int = Field(description="Predicted class (0=benign, 1=suspicious, 2=malicious)")
    probabilities: List[float] = Field(description="Class probabilities")
    confidence: float = Field(description="Prediction confidence")
    uncertainty: float = Field(description="Prediction uncertainty")
    threat_level: str = Field(description="Human-readable threat level")
    explanations: Dict = Field(description="Explanation details")
    validation: Dict = Field(description="Validation metrics")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model."""
    samples: List[ThreatSample]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""
    predictions: List[ThreatPrediction]
    summary: Dict[str, float]


# Initialize FastAPI app
app = FastAPI(
    title="Cyber Threat AI Detection API",
    description="AI-powered cybersecurity threat detection with neuro-symbolic reasoning",
    version="0.1.0"
)

# Global inference engine
inference_engine: Optional[object] = None


@app.on_event("startup")
async def startup_event():
    """Initialize inference engine on startup."""
    global inference_engine
    config = get_config()
    
    # Load best model checkpoint
    model_path = "checkpoints/best_model.pt"
    
    try:
        inference_engine = create_inference_engine(model_path)
        logger.info("Inference engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Cyber Threat AI Detection API",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": inference_engine is not None
    }


@app.post("/predict", response_model=ThreatPrediction)
async def predict_threat(sample: ThreatSample):
    """Predict threat for a single sample.
    
    Args:
        sample: Input threat sample
        
    Returns:
        Threat prediction result
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    try:
        # Convert to dictionary
        sample_dict = sample.dict(by_alias=True, exclude_unset=True)
        
        # Make prediction
        result = inference_engine.predict_single(sample_dict)
        
        return ThreatPrediction(**result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_threats_batch(request: BatchPredictionRequest):
    """Predict threats for multiple samples.
    
    Args:
        request: Batch prediction request
        
    Returns:
        Batch prediction results
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    try:
        # Convert samples to dictionaries
        sample_dicts = [sample.dict(by_alias=True, exclude_unset=True) for sample in request.samples]
        
        # Make predictions
        results = inference_engine.predict_batch(sample_dicts)
        
        # Convert to response models
        predictions = [ThreatPrediction(**result) for result in results]
        
        # Calculate summary statistics
        threat_counts = {0: 0, 1: 0, 2: 0}
        total_confidence = 0.0
        total_uncertainty = 0.0
        
        for pred in predictions:
            threat_counts[pred.prediction] += 1
            total_confidence += pred.confidence
            total_uncertainty += pred.uncertainty
        
        summary = {
            "total_samples": len(predictions),
            "benign_count": threat_counts[0],
            "suspicious_count": threat_counts[1], 
            "malicious_count": threat_counts[2],
            "avg_confidence": total_confidence / len(predictions),
            "avg_uncertainty": total_uncertainty / len(predictions)
        }
        
        return BatchPredictionResponse(predictions=predictions, summary=summary)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# Main entry points
def create_app(config_path: Optional[str] = None) -> FastAPI:
    """Create FastAPI application.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured FastAPI application
    """
    return app


def run_server(host: str = "0.0.0.0", port: int = 8080, **kwargs):
    """Run the FastAPI server.
    
    Args:
        host: Server host
        port: Server port
        **kwargs: Additional uvicorn arguments
    """
    uvicorn.run(app, host=host, port=port, **kwargs)


if __name__ == "__main__":
    config = get_config()
    run_server(
        host=config.get("serve.host", "0.0.0.0"),
        port=config.get("serve.port", 8080)
    )
