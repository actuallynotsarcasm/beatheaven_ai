FROM python:3.10
WORKDIR /
RUN apt-get update
RUN apt-get install -y ffmpeg
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
COPY app /app
EXPOSE 8000
CMD cd app && python app.py