import os
import json
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage

# Carica variabili d'ambiente
load_dotenv()

# 1. Configurazione del Modello (OpenRouter)
model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="gpt-4o-mini",
    temperature=0.1,
)

# 2. Tool di Parsing Puro
@tool
def get_all_parsed_sms() -> str:
    """
    Reads the sms.json file and parses each raw SMS entry from its unstructured text block 
    into a structured format. 
    
    Returns: a list containing the sender, date, and message content for every SMS,
    """
    try:
        with open('sms.json', 'r') as f:
            data = json.load(f)
        
        parsed_list = []
        # Regex basata sul formato ufficiale: From, To, Date, Message 
        pattern = r"From: (.*)\nTo: (.*)\nDate: (.*)\nMessage: (.*)"
        
        for entry in data:
            raw_text = entry.get("sms", "")
            match = re.search(pattern, raw_text, re.DOTALL)
            if match:
                parsed_list.append({
                    "mittente": match.group(1).strip(),
                    "data": match.group(3).strip(),
                    "testo": match.group(4).strip()
                })
        
        return json.dumps(parsed_list, indent=2)
    
    except FileNotFoundError:
        return "Errore: file sms.json non trovato."
    except Exception as e:
        return f"Errore durante il parsing: {str(e)}"

#agent creation
sms_agent = create_agent(
    model=model,
    system_prompt="When user asks for sms evaluation, use the provided tool to parse the sms.json file." \
    "Give to each text a score form 0 to 1 based on the presence of scam indicators such as urgency in the message." \
    "Return the list of sender, recipient, date, and score for each sms.",
    tools=[get_all_parsed_sms]
)
