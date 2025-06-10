# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies including Microsoft ODBC Driver
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    unixodbc-dev \
    && curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /usr/share/keyrings/microsoft-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft-archive-keyring.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy entire app
COPY . .

# Create .streamlit directory and config
RUN mkdir -p .streamlit

# Create Streamlit config file for HTTPS
RUN echo '[server]\n\
    port = 443\n\
    enableCORS = false\n\
    enableXsrfProtection = false\n\
    sslCertFile = "/app/cert.pem"\n\
    sslKeyFile = "/app/key.pem"\n\
    headless = true\n\
    ' > .streamlit/config.toml

# Set Python path
ENV PYTHONPATH=/app

# Expose port 443 for HTTPS
EXPOSE 443

# Run Streamlit with HTTPS configuration
CMD ["streamlit", "run", "app/reportings/mcc_report.py", "--server.port=443", "--server.address=0.0.0.0"]