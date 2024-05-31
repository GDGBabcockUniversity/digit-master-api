docker run -p 8501:8501 --name=tf_serving \
  --mount type=bind,source=$(pwd)/model,target=/models/digit_master \
  -e MODEL_NAME=digit_master -t tensorflow/serving
