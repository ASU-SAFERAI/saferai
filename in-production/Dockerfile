ARG LAMBDA_TASK_ROOT=/var/task

FROM public.ecr.aws/lambda/python:3.12 as build-image

ENV LAMBDA_TASK_ROOT=${LAMBDA_TASK_ROOT}
WORKDIR ${LAMBDA_TASK_ROOT}

ADD . ${LAMBDA_TASK_ROOT}

COPY requirements.txt .
RUN pip install --no-cache-dir --target ${LAMBDA_TASK_ROOT} -r requirements.txt

# RUN find ${LAMBDA_TASK_ROOT} -name "*.pyc" -delete
# RUN find ${LAMBDA_TASK_ROOT} -name "__pycache__" -delete

# RUN python -m compileall ${LAMBDA_TASK_ROOT}

FROM python:3.12-slim

ARG LAMBDA_TASK_ROOT

ENV LAMBDA_TASK_ROOT=${LAMBDA_TASK_ROOT}
WORKDIR ${LAMBDA_TASK_ROOT}

COPY --from=build-image ${LAMBDA_TASK_ROOT} .

ENTRYPOINT ["/usr/local/bin/python", "-m", "awslambdaric"]
CMD ["main.lambda_handler"]
