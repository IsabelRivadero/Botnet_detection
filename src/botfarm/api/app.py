from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .routes import router

app = FastAPI(
    title="BotFarm API",
    description="API REST para detección de granjas de bots, análisis point-in-time y visualización",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/docs")

# Montar carpeta de reportes estáticos (HTMLs y PNGs)
reports_dir = Path(__file__).resolve().parents[3] / "reports"
reports_dir.mkdir(exist_ok=True)
app.mount("/reports", StaticFiles(directory=str(reports_dir)), name="reports")

app.include_router(router, prefix="/api/v1")