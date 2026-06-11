from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.upload import router as upload_router
from app.routes.ask import router as ask_router
from app.services.vector_store import load_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once before the server accepts any requests.
    # Restores the FAISS index from disk so search works immediately
    # without re-uploading documents after every restart.
    load_index()
    yield


app = FastAPI(title="Document Q&A API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, prefix="/api", tags=["Upload"])
app.include_router(ask_router, prefix="/api", tags=["Ask"])


@app.get("/health")
def health():
    return {"status": "ok"}
