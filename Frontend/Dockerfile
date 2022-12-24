FROM python:3.10.8


COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ENV IS_DOCKERIZED true
COPY . .

CMD ["streamlit", "run", "Home.py","--browser.gatherUsageStats","False"]
