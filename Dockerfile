From python:latest

WORKDIR /app/ChatPdf/

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8501

CMD streamlit run app.py 