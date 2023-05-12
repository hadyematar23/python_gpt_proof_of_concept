import pdb
import os 
import pickle
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, download_loader, GPTListIndex, LLMPredictor, PromptHelper
from llama_index import ServiceContext, StorageContext, load_index_from_storage
from langchain.chat_models import ChatOpenAI

os.environ['OPENAI_API_KEY'] = 'INSERT_API_KEY'

max_input_size = 4096
num_output = 256
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
llm_predictor_gpt = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_output))

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor_gpt, prompt_helper=prompt_helper)

def authorize_gdocs():
	google_oauth2_scopes = [
		"https://www.googleapis.com/auth/documents.readonly"
	]
	cred = None
	if os.path.exists("token.pickle"):
		with open("token.pickle", 'rb') as token:
			cred = pickle.load(token)
	if not cred or not cred.valid:
		if cred and cred.expired and cred.refresh_token:
			cred.refresh(Request())
		else:
			flow = InstalledAppFlow.from_client_secrets_file("credentials.json", google_oauth2_scopes)
			cred = flow.run_local_server(port=3000)
		with open("token.pickle", 'wb') as token:
			pickle.dump(cred, token)


# function to authorize or download latest credentials 
authorize_gdocs()

# initialize LlamaIndex google doc reader
GoogleDocsReader = download_loader('GoogleDocsReader')

# list of google docs we want to index 
gdoc_ids = ['1h88X1fim6qGuofDqwn0Kf3_zekWIpFO5JjFaMQxz2QQ', '1cIyjKwJ1HNLtQ1UNwFqgb8dycJ4v0RoK_rGl6j6mrWo', '1OYyXhZ3vZt9_hN-_PLRYwAEaghAchk7S-vdoY4B_jVs']

loader = GoogleDocsReader()

# load gdocs and index them 
documents = loader.load_data(gdoc_ids)

index = GPTVectorStoreIndex.from_documents(
    documents, service_context=service_context
)
index.storage_context.persist(persist_dir = 'storage')


storage_context = StorageContext.from_defaults(persist_dir = 'storage')
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()

response = query_engine.query("With regards to the right to terminate an abortion, can you discuss how these three cases are related? Can you tell me which one is controlling precedent as of December 2022? In your response, it important to base your responses strictly on the cases I provide you. If a case within this set refers to another case not included, you may reference it as part of your analysis. However, please don't rely on any cases outside the corpus of cases I have provided or use any foundational knowledge regarding the specifics of any cases. All technical knowledge of the cases should come only from the corpus of information (the cases) that I have provided you. You may utilize your foundational knowledge regarding how precedence works.")

pdb.set_trace()