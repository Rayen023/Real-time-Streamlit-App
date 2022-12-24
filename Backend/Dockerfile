FROM python:3.10.8


COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "mlapi:app", "--reload"]

