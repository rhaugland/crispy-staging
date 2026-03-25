from fastapi import FastAPI

from api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Virtual Staging API",
        version="0.1.0",
        description="Self-service virtual staging for real estate photos",
    )
    app.include_router(router)
    return app
