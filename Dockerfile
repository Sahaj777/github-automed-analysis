FROM python:3.10
ENV PYTHONUNBUFFERED True
WORKDIR /main
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install flask
COPY . /main
RUN pip install transformers
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch torchvision transformers
RUN pip install gunicorn
RUN pip install aiohttp
RUN pip install asynction
RUN pip install flask[async]

# RUN python -c "from transformers import GPT2LMHeadModel, GPT2Tokenizer; tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium'); model = GPT2LMHeadModel.from_pretrained('gpt2-medium');"

EXPOSE 3000
CMD ["gunicorn", "--bind", "0.0.0.0:3000", "main:app"]

