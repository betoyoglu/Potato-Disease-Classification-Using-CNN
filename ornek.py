from fastapi import FastAPI

app= FastAPI()

@app.get("/hello")
async def hello():
    return "welcome"

#çalıştırmak için >>  uvicorn main:app --reload

## fast api ile birlikte tf server kullanıcaz >> tf serving makes model 
# version management and model serving very easy

##localhost:8000/docs >> swagger gibi
