FROM anibali/pytorch:1.8.1-cuda11.1

RUN sudo apt-get update
RUN sudo apt-get install -y vim
RUN sudo apt-get install -y screen
RUN sudo apt-get install -y gcc
RUN sudo apt-get install -y wget

RUN pip install numpy
RUN pip install scikit-learn
RUN pip install cvxpy
RUN pip install coffea
RUN pip install awkward

RUN sudo apt-get update

# Set the default command to python3.
CMD ["python3"]
