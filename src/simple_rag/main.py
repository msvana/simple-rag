import fastapi

from . import api

app = fastapi.FastAPI()
app.include_router(api.router)


