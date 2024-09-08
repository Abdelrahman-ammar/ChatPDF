# Chat PDF using Gemini

This project is just as the name suggests , you are talking with your pdfs using Gemini LLM from google in addition to Memory which is provided from the popular library langchain, however there is some steps which is a must before running this project.

## How to start (Repo Clone)

---

1. **Cloning the repo** : git clone this repo
2. Navigate to the project directory.
3. **Install requirement.txt file** that contains the dependencies :
   `pip install -r requirements.txt`

4. Set up your environment variables by creating a .env file in the project directory and adding your Google API key:
   `GOOGLE_API_KEY=your_api_key_here`

5. **Run streamlit app.py** : `streamlit run app.py`

to get your API Key you can visit : https://makersuite.google.com/app/apikey

## Docker startup:

---

1. A `Dockerfile` is provided which can be used for deployment. From this `Dockerfile` a docker image can be created and deployed in cloud, etc.

2. Then, open a command line cmd at the root of the repository, and run the command: `docker build -t chatpdf:v1.0 .`

3. Once the image is created , build the container using `docker run -d -p 8501:8501 chatpdf:v1.0`

In both clonning and dockerfile startup you will have to navigate to your web browser to interact with the project.

## How to use:

1. After accessing the web browser you will have to browse , select and submit the pdf.

2. Once the processing is completed and the knowledge base is created , you can ask questions related to the content you provided

However there is some limitations , which will be adhered in the futrue , like only files with pdf extensions can be uploaded and processed , however this also require a naive pdf where the file should not contain any images or structures like table or other shapes, (again this is only version 1 and will be enhanced in the feature).
