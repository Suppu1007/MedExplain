from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.imaging_api import router as imaging_router

app = FastAPI(title="MediExplain Imaging Module")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(imaging_router, prefix="/api/imaging", tags=["Imaging"])
