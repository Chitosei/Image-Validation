<h1>Setup and Installation</h1>

This guide will help you set up and run the Image-Validation project.

### 1. Clone the Repository

```bash
git clone https://github.com/Chitosei/Image-Validation.git
cd src
```

### 2. Install Python Dependencies 
Install the required Python libraries using the ```requirements.txt``` file:
```bash
pip install -r requirements.txt
```

### 3. Run the Backend
Start the backend server using Uvicorn:

```bash 
uvicorn main:app --reload
```

### 4. Accessing the FastAPI Documentation


Open your web browser and navigate to the following URL to access the FastAPI documentation:

```http://127.0.0.1:8000/docs```

This will display the interactive documentation for your FastAPI application.

### 5. Setting Up Ngrok (Optional)
Note: This step is optional if you only intend to run the application locally. Ngrok allows you to expose your local server to the public internet for testing purposes.

For Windows users with PowerShell, follow these steps to configure Ngrok with an authtoken:

```PowerShell
ngrok config add-authtoken 2qjW77o8y7OwHjXUnrj9MN40XkU_3JJNRnSFDhxroDNrjo23S
```
<strong>Important: Replace ```2qjW77o8y7OwHjXUnrj9MN40XkU_3JJNRnSFDhxroDNrjo23S``` with your own Ngrok authtoken. You can obtain your authtoken from the Ngrok dashboard.
