FROM python:3.10
WORKDIR /
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN apt-get install ffmpeg
COPY app /app
EXPOSE 8000
CMD cd app && python app.py