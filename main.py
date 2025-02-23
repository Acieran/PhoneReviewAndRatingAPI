from fastapi import FastAPI
import kagglehub

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}



# Download latest version
path = kagglehub.dataset_download("jacksondivakarr/phone-classification-dataset")

print("Path to dataset files:", path)


