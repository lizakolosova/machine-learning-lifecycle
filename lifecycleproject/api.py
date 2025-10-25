import joblib
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
import pandas as pd
import traceback
from fastapi.responses import PlainTextResponse

final_pipeline = joblib.load("artifacts/final_mushroom_pipeline.pkl")
prep_pkg = joblib.load("artifacts/mushroom_preprocessing_pipeline.pkl")
target_encoder = prep_pkg["target_encoder"]

app = FastAPI(
    title="Mushroom Classification API",
    description="Predict whether a mushroom is edible or poisonous.",
    version="1.0"
)


class MushroomFeatures(BaseModel):
    cap_shape: str = Field(alias="cap-shape")
    cap_surface: str = Field(alias="cap-surface")
    cap_color: str = Field(alias="cap-color")
    bruises: str
    odor: str
    gill_attachment: str = Field(alias="gill-attachment")
    gill_spacing: str = Field(alias="gill-spacing")
    gill_size: str = Field(alias="gill-size")
    gill_color: str = Field(alias="gill-color")
    stalk_shape: str = Field(alias="stalk-shape")
    stalk_root: str = Field(alias="stalk-root")
    stalk_surface_above_ring: str = Field(alias="stalk-surface-above-ring")
    stalk_surface_below_ring: str = Field(alias="stalk-surface-below-ring")
    stalk_color_above_ring: str = Field(alias="stalk-color-above-ring")
    stalk_color_below_ring: str = Field(alias="stalk-color-below-ring")
    veil_type: str = Field(alias="veil-type")
    veil_color: str = Field(alias="veil-color")
    ring_number: str = Field(alias="ring-number")
    ring_type: str = Field(alias="ring-type")
    spore_print_color: str = Field(alias="spore-print-color")
    population: str
    habitat: str

    class Config:
        allow_population_by_alias = True


@app.post("/predict")
def predict_mushroom(features: MushroomFeatures):
    try:
        X_input = pd.DataFrame([features.model_dump(by_alias=True)])

        prediction = final_pipeline.predict(X_input)
        label = str(target_encoder.inverse_transform(prediction)[0])
        return {"prediction": label}
    except Exception as e:
        return PlainTextResponse(
            content=f"Error: {e}\n\n{traceback.format_exc()}",
            status_code=500
        )


@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        tb = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        print("==== Exception caught ====")
        print(tb, flush=True)
        return PlainTextResponse(content=tb, status_code=500)
