from fastapi import FastAPI, File, UploadFile 
from fastapi.responses import JSONResponse 
import uvicorn 
 
app = FastAPI(title="Architectural Style Classification API") 
 
@app.get("/") 
async def root(): 
    return {"message": "Architectural Style Classification API is running!"} 
 
@app.get("/health") 
async def health(): 
    return {"status": "healthy"} 
 
@app.post("/predict") 
async def predict(file: UploadFile = File(...)): 
    return { 
        "filename": file.filename, 
        "prediction": "test_prediction", 
        "confidence": 0.95 
    } 
 
if __name__ == "__main__": 
    uvicorn.run(app, host="0.0.0.0", port=8000) 
