# 1. Use the official lightweight Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install packages directly 
# (In production we use requirements.txt, but inline is cleaner for the students' first isolated project)
RUN pip install --no-cache-dir flask pandas numpy scipy scikit-learn

# 4. Copy the Application Code
COPY app.py .

# 5. Expose port 7860 (Hugging Face standard)
EXPOSE 7860

# 6. Run the Flask Webserver securely
CMD ["flask", "--app", "app.py", "run", "--host=0.0.0.0", "--port=7860"]
