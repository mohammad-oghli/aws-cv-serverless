FROM public.ecr.aws/lambda/python:3.9

COPY config.py helper.py model.py server.py requirements.txt ./
RUN python3.9 -m pip install -r requirements.txt -t .

COPY model /ml/model/
COPY utils utils/

CMD ["server.lambda_handler"]
