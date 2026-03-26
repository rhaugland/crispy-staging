from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Virtual Staging API",
        version="0.1.0",
        description="Self-service virtual staging for real estate photos",
    )
    app.include_router(router)

    # Serve static frontend
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(static_dir / "index.html"))

    return app


app = create_app()
