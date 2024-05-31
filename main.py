from fastapi import FastAPI, Response, status
from nltk.tokenize import word_tokenize
from fastapi.middleware.cors import CORSMiddleware
from qdrant_query import Query
import uvicorn


app = FastAPI()
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", status_code=200)
async def retrieve_data(response: Response, query_str: str = "", maxToken = 512):
    print("queries: ", query_str)
    if query_str.strip() != "":
        token = 0
        result = []
        for res in Query(query_str):
            if token >= maxToken:
                break
            description: str = res.payload["description"]
            token += len(word_tokenize(description))
            result.append(description)
        return result
    else:
        print("Please provide query_str in the parameters")
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {
            "msg": "Please provide query_str in the parameters"
        }
        
if __name__ == '__main__':
    uvicorn.run(app, port=8888)