# Use official Python 3.11 image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port Render will use
EXPOSE 10000

# Start the app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]
