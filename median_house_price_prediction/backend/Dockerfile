# Pyhon image
FROM python:3.10

# set working folder inside the container
WORKDIR /app

# copy only requirements first ( This helps Docker cache installs faster)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy everything  	except (main.py, model file, etc.)
COPY . .

# expose port(Tell docker which port the app uses)
EXPOSE 8000

# run the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
