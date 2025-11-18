# Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask.
#clone the repository
base
git clone https://github.com/SUJITHA0312/Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask.

#step 1: create a conda environment after opening the repository
bash
conda create -n medibot python=3.11 -y

bash
conda activate medibot

#step2: install the requirements
bash
pip install -r requirements.txt

Create a .env file in the root directory and add your Pinecone & openai credentials as follows:
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# run the following command to store embeddings to pinecone
python store_index.py

# Finally run the following command
python app.py

Now,

open up localhost:
Techstack Used:
Python
LangChain
Flask
GPT
Pinecone
ECR_REPO
PINECONE_API_KEY
OPENAI_API_KEY
