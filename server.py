import numpy as np
from io import BytesIO
from fastapi import FastAPI
import matplotlib.pyplot as plt
import matplotlib as mpl
from starlette.responses import StreamingResponse


X = np.load('full_numpy_bitmap_alarm clock.npy', encoding='latin1', allow_pickle=True)

app = FastAPI()

@app.get("/generate/{idx}")
async def generate(idx: int):
    mpl.use('Agg')
    fig = plt.figure()
    plt.imshow(X[idx].reshape(28, 28), cmap='Greys')

    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return StreamingResponse(content=buf, media_type="image/png")
