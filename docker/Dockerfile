FROM yidarvin/bmcv-base:latest

# Project Specific
RUN pip install tqdm
RUN pip install nibabel
RUN pip install pydicom

# Add Path
ENV PATH="/home/bmcv/brainmets:${PATH}"
ENV PYTHONPATH="/home/bmcv/brainmets:${PYTHONPATH}"

CMD ["bash"]