import os
import json
import re
import ulid
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler

import math
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from pydantic import BaseModel, Field

import numpy as np
from collections import Counter
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

# Carica variabili d'ambiente
load_dotenv()

# 1. Configurazione del Modello (OpenRouter)
model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="gpt-4o-mini",
    temperature=0.1,
)

# 2. Inizializzazione del contesto condiviso per i tool
_context: dict = {
    "transaction": {},
    "history": [],
    "user_profile": {}
}

# ===========================================================================
# TOOLS: COMMUNICATION & PROFILE
# ===========================================================================

@tool
def get_all_parsed_sms() -> str:
    """Reads the sms.json file and parses each raw SMS entry from its unstructured text block into a structured format."""
    try:
        with open('sms.json', 'r') as f:
            data = json.load(f)
        
        parsed_list = []
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

@tool
def get_user_profile() -> str:
    """Restituisce il profilo demografico e le abitudini dell'utente."""
    profile = _context.get("user_profile", {})
    if not profile:
        return "Profilo utente non disponibile."
    return (
        f"Nome: {profile.get('first_name')} {profile.get('last_name')}\n"
        f"Lavoro: {profile.get('job')}\n"
        f"Stipendio annuo: €{profile.get('salary')}\n"
        f"Città: {profile.get('residence', {}).get('city', 'N/A')}\n"
        f"Descrizione: {profile.get('description', 'N/A')}"
    )

@tool
def get_user_history(n: int = 5) -> str:
    """Restituisce le ultime N transazioni dell'utente."""
    history = _context.get("history", [])
    window  = history[-n:] if len(history) >= n else history
    if not window:
        return "Nessuna transazione precedente disponibile."
    lines = []
    for tx in window:
        lines.append(
            f"[{str(tx.get('timestamp', ''))[:10]}] tipo={tx.get('transaction_type', 'N/A')} "
            f"importo={tx.get('amount', 0)} iban_dest={tx.get('recipient_iban', 'N/A')} "
            f"desc={tx.get('description', 'N/A')} balance={tx.get('balance_after', 'N/A')}"
        )
    return "\n".join(lines)

@tool
def compute_behavioral_stats() -> str:
    """Calcola statistiche comportamentali dell'utente: media/std importi, tipi, orari, IBAN frequenti."""
    history = _context.get("history", [])
    if not history: return "Storia insufficiente."
    amounts = [float(tx["amount"]) for tx in history if "amount" in tx]
    types   = [tx["transaction_type"] for tx in history if "transaction_type" in tx]
    hours   = [datetime.fromisoformat(str(tx["timestamp"])).hour for tx in history if "timestamp" in tx]
    ibans   = [tx.get("recipient_iban", "") for tx in history if tx.get("recipient_iban")]
    
    if not amounts: return "Nessun importo valido."

    stats = {
        "importo_medio": round(float(np.mean(amounts)), 2),
        "importo_std": round(float(np.std(amounts)), 2),
        "importo_min": round(float(np.min(amounts)), 2),
        "importo_max": round(float(np.max(amounts)), 2),
        "tipi_tx_usati": dict(Counter(types)),
        "ora_media_tx": round(float(np.mean(hours)), 1) if hours else 0.0,
        "ora_std_tx": round(float(np.std(hours)), 1) if hours else 0.0,
        "recipient_ibans_frequenti": dict(Counter(ibans)),
        "totale_tx_storia": len(history),
    }
    return json.dumps(stats, ensure_ascii=False, indent=2)

@tool
def check_recipient_known() -> str:
    """Verifica se il destinatario IBAN è già noto."""
    history = _context.get("history", [])
    current_iban = _context.get("transaction", {}).get("recipient_iban", "")
    if not current_iban: return "Nessun IBAN destinatario."
    occurrences = sum(1 for tx in history if tx.get("recipient_iban", "") == current_iban)
    if occurrences == 0:
        return f"IBAN {current_iban} MAI visto prima. Destinatario nuovo."
    return f"IBAN {current_iban} già visto {occurrences} volta/e. Destinatario noto."

# ===========================================================================
# TOOLS: LOCATION (GEOSPATIAL)
# ===========================================================================

def _load_data():
    locs = pd.DataFrame(json.loads(Path("locations.json").read_text()))
    locs["timestamp"] = pd.to_datetime(locs["timestamp"])
    
    txns = pd.read_csv("transactions.csv")
    txns["timestamp"] = pd.to_datetime(txns["timestamp"])
    
    users = pd.DataFrame(json.loads(Path("users.json").read_text()))
    
    iban_to_biotag = {}
    for _, row in txns.iterrows():
        if pd.notna(row.get("sender_iban")) and pd.notna(row.get("sender_id")):
            iban_to_biotag[row["sender_iban"]] = row["sender_id"]
        if pd.notna(row.get("recipient_iban")) and pd.notna(row.get("recipient_id")):
            iban_to_biotag[row["recipient_iban"]] = row["recipient_id"]
    
    users["biotag"] = users["iban"].map(iban_to_biotag)
    return locs, txns, users

_locs, _txns, _users = _load_data()

def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

class UserCoherencyInput(BaseModel):
    biotag: str = Field(description="User biotag")
    window_hours: int = Field(6, description="Hours around transaction")
    max_plausible_kmh: float = Field(900.0, description="Max speed km/h")

class AllUsersCoherencyInput(BaseModel):
    window_hours: int = Field(6, description="Hours around transaction")
    max_plausible_kmh: float = Field(900.0, description="Max speed km/h")

@tool(args_schema=UserCoherencyInput)
def get_user_coherency_score(biotag: str, window_hours: int = 6, max_plausible_kmh: float = 900.0) -> dict:
    """Calculate coherency score (0-1) for a single user based on GPS-transaction alignment."""
    user = _users[_users["biotag"] == biotag]
    if user.empty: return {"error": f"User {biotag} not found"}
    
    home_lat = float(user.iloc[0]["residence"]["lat"])
    home_lng = float(user.iloc[0]["residence"]["lng"])
    user_txns = _txns[(_txns["sender_id"] == biotag) | (_txns["recipient_id"] == biotag)]
    if user_txns.empty: return {"biotag": biotag, "score": 1.0, "reason": "No txns"}
    
    user_locs = _locs[_locs["biotag"] == biotag]
    violations, total_weight = 0, 0
    
    for _, txn in user_txns.iterrows():
        txn_time = txn["timestamp"]
        nearby = user_locs[abs(user_locs["timestamp"] - txn_time) <= timedelta(hours=window_hours)]
        
        weight = 0.3 if txn["transaction_type"] == "transfer" else 0.1 if txn["transaction_type"] in ["e-commerce", "direct debit"] else 1.0
        total_weight += weight
        
        if nearby.empty:
            violations += weight * 0.5
            continue
        
        for _, gps in nearby.iterrows():
            dist = _haversine_km(gps["lat"], gps["lng"], home_lat, home_lng)
            time_diff_h = abs((gps["timestamp"] - txn_time).total_seconds()) / 3600
            if time_diff_h > 0:
                if (dist / time_diff_h) > max_plausible_kmh and dist > 10:
                    violations += weight
                    break
            elif txn["transaction_type"] == "in-person payment" and dist > 100:
                violations += weight * 0.8
                break
    
    score = max(0.0, 1.0 - (violations / total_weight if total_weight > 0 else 0))
    return {"coherency_score": round(score, 3), "interpretation": "High" if score > 0.8 else "Low"}

@tool(args_schema=AllUsersCoherencyInput)
def get_all_users_coherency(window_hours: int = 6, max_plausible_kmh: float = 900.0) -> dict:
    """Calculate coherency scores (0-1) for all users in the system."""
    results = [get_user_coherency_score.invoke({"biotag": b}) for b in _users["biotag"].dropna().unique()]
    valid_results = [r for r in results if "error" not in r]
    return {"average_score": sum(r["coherency_score"] for r in valid_results) / max(len(valid_results), 1)}

@tool
def get_user_transaction_details(biotag: str) -> dict:
    """Get detailed transaction history with location context for a specific user."""
    return {"status": "implement full payload logic if queried"}

# ===========================================================================
# SUB-AGENTS CREATION
# ===========================================================================

sms_agent = create_agent(
    model=model,
    system_prompt="""You are a specialized cybersecurity agent analyzing SMS communications.
    When user asks for sms evaluation, use the provided tool to parse the sms.json file.
    Give to each text a score form 0 to 1 based on the presence of scam indicators such as urgency in the message.""",
    tools=[get_all_parsed_sms]
)

profile_agent = create_agent(
    model=model,
    system_prompt=r"""Sei un agente specializzato nel rilevamento di frodi bancarie analizzando il profilo finanziario.
    Usa i tool per raccogliere dati storici e calcolare lo Z-score. 
    Restituisci un'analisi dettagliata in formato JSON con 'score' e 'metrics'.""",
    tools=[get_user_profile, get_user_history, compute_behavioral_stats, check_recipient_known]
)

location_agent = create_agent(
    model=model,
    system_prompt=r"""Sei un agente specializzato nell'analisi geospaziale e della coerenza GPS per il rilevamento frodi.
    Utilizza i tool a tua disposizione per verificare incroci spazio-temporali, velocità di spostamento impossibili e distanza dall'indirizzo di residenza.""",
    tools=[get_user_coherency_score, get_all_users_coherency, get_user_transaction_details]
)

# ===========================================================================
# MASTER AGENT & DELEGATION TOOLS
# ===========================================================================

@tool
def delegate_to_profile_agent(query: str) -> str:
    """Delega all'esperto finanziario l'analisi del profilo e delle anomalie transazionali."""
    res = profile_agent.invoke({"input": query})
    return str(res.get("output", res))

@tool
def delegate_to_sms_agent(query: str) -> str:
    """Delega all'esperto di cybersecurity la ricerca di pattern di phishing o ingegneria sociale."""
    res = sms_agent.invoke({"input": query})
    return str(res.get("output", res))

@tool
def delegate_to_location_agent(query: str) -> str:
    """Delega all'esperto geospaziale la verifica della coerenza GPS e l'analisi cinetica dell'utente."""
    res = location_agent.invoke({"input": query})
    return str(res.get("output", res))

master_agent = create_agent(
    model=model,
    system_prompt=r"""Sei il Master Classifier Agent, un Senior Fraud Analyst e coordinatore.
    Riceverai in input i dati JSON grezzi di una transazione e del dataset.
    
    Il tuo compito NON è processare direttamente tutto, ma DELEGARE il lavoro ai tuoi sub-agenti specializzati usando i tuoi tool:
    1. Usa 'delegate_to_profile_agent' passando i dettagli della transazione per lo Z-score e il profilo.
    2. Usa 'delegate_to_sms_agent' per far verificare se l'utente è stato compromesso via messaggio.
    3. Usa 'delegate_to_location_agent' passando il biotag per validare la coerenza geografica.
    
    Attendi e analizza le loro risposte incrociando i dati (es. Importo Anomalo + GPS non coerente = Rischio Altissimo).
    
    Produci ESCLUSIVAMENTE questo JSON alla fine della tua analisi:
    {
      "final_fraud_score": <float 0.0-1.0>,
      "is_fraud": <bool>,
      "master_reasoning": "<spiegazione in italiano su come hai incrociato i report dei 3 agenti>"
    }""",
    tools=[delegate_to_profile_agent, delegate_to_sms_agent, delegate_to_location_agent]
)

# ===========================================================================
# UTILS & LANGFUSE
# ===========================================================================

langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")
)

def generate_session_id():
    team = os.getenv("TEAM_NAME", "tutorial").replace(" ", "-")
    return f"{team}-{ulid.new().str}"

def invoke_langchain(model, prompt, langfuse_handler, session_id):
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(
        messages,
        config={
            "callbacks": [langfuse_handler],
            "metadata": {"langfuse_session_id": session_id},
        },
    )
    return response.content

@observe()
def run_llm_call(session_id, agent, prompt):
    langfuse_handler = CallbackHandler()
    response = invoke_langchain(agent, prompt, langfuse_handler, session_id)
    return response

print("✓ Langfuse initialized successfully")
print("✓ Multi-Agent system ready (sms_agent, profile_agent, location_agent, master_agent)")

# ===========================================================================
# LOAD AND PREPARE DATA FOR ANALYSIS
# ===========================================================================

# Load the full dataset
txns_df = pd.read_csv("transactions.csv")
txns_df["timestamp"] = pd.to_datetime(txns_df["timestamp"])

# Load user profiles for context
with open("users.json", "r") as f:
    users_data = json.load(f)

# Create biotag to user mapping
biotag_to_user = {}
iban_to_biotag = {}
for _, row in txns_df.iterrows():
    if pd.notna(row.get("sender_iban")) and pd.notna(row.get("sender_id")):
        iban_to_biotag[row["sender_iban"]] = row["sender_id"]
    if pd.notna(row.get("recipient_iban")) and pd.notna(row.get("recipient_id")):
        iban_to_biotag[row["recipient_iban"]] = row["recipient_id"]

for user in users_data:
    biotag = iban_to_biotag.get(user["iban"])
    if biotag:
        biotag_to_user[biotag] = user
        user["biotag"] = biotag

# Get unique biotags from users
active_biotags = [u["biotag"] for u in users_data if "biotag" in u]

# Prepare sample transactions for analysis (take a few interesting ones)
sample_txns = txns_df[
    (txns_df["sender_id"].isin(active_biotags)) | 
    (txns_df["recipient_id"].isin(active_biotags))
].head(10)  # Limit to 10 for demo

# Create context for each transaction to analyze
analysis_prompts = []
for _, txn in sample_txns.iterrows():
    biotag = txn["sender_id"] if pd.notna(txn["sender_id"]) else txn["recipient_id"]
    user = biotag_to_user.get(biotag, {})
    
    # Set the global context for tools that need it
    _context["transaction"] = txn.to_dict()
    _context["user_profile"] = user
    
    # Get user's transaction history
    user_history = txns_df[
        (txns_df["sender_id"] == biotag) | 
        (txns_df["recipient_id"] == biotag)
    ].sort_values("timestamp")
    _context["history"] = user_history.to_dict(orient="records")

# ===========================================================================
# ANALYZE ALL TRANSACTIONS
# ===========================================================================

print("\n" + "=" * 80)
print("FRAUD ANALYSIS REPORT")
print("=" * 80)

for idx, (_, txn) in enumerate(sample_txns.iterrows(), 1):
    biotag = txn["sender_id"] if pd.notna(txn["sender_id"]) else txn["recipient_id"]
    user = biotag_to_user.get(biotag, {})
    
    # Update context for this transaction
    _context["transaction"] = txn.to_dict()
    _context["user_profile"] = user
    user_history = txns_df[
        (txns_df["sender_id"] == biotag) | 
        (txns_df["recipient_id"] == biotag)
    ].sort_values("timestamp")
    _context["history"] = user_history.to_dict(orient="records")
    
    # Create analysis prompt with all necessary data
    analysis_prompt = f"""
Analyze this transaction for fraud risk. Here's the transaction data:
{json.dumps(txn.to_dict(), default=str, indent=2)}

User profile:
{json.dumps(user, indent=2) if user else 'Unknown user'}

Transaction history for this user (last 5):
{json.dumps([t for t in _context['history'][-5:]], default=str, indent=2)}

Please:
1. Delegate to profile_agent to check behavioral anomalies
2. Delegate to sms_agent to check for phishing indicators
3. Delegate to location_agent with biotag '{biotag}' to check GPS coherence
4. Combine findings into a final fraud score (0.0-1.0)

Output ONLY valid JSON with: final_fraud_score, is_fraud, master_reasoning
"""
    
    print(f"\n📊 Analyzing Transaction #{idx}: {txn['transaction_id'][:8]}...")
    print(f"   User: {user.get('first_name', 'Unknown')} {user.get('last_name', '')}")
    print(f"   Amount: €{txn['amount']} | Type: {txn['transaction_type']}")
    
    try:
        # Invoke master agent with full context
        response = master_agent.invoke({
            "messages": [HumanMessage(content=analysis_prompt)]
        })
        
        # Extract the final response
        final_output = response["messages"][-1].content
        
        # Try to parse as JSON
        try:
            # Find JSON in response (in case there's extra text)
            json_match = re.search(r'\{.*\}', final_output, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                print(f"   ✓ Fraud Score: {result.get('final_fraud_score', 'N/A')}")
                print(f"   ✓ Is Fraud: {result.get('is_fraud', 'N/A')}")
                print(f"   ✓ Reasoning: {result.get('master_reasoning', 'N/A')[:100]}...")
            else:
                print(f"   ⚠️ Could not parse JSON from response")
                print(f"   Raw response: {final_output[:200]}...")
        except json.JSONDecodeError as e:
            print(f"   ⚠️ JSON parse error: {e}")
            print(f"   Raw response: {final_output[:200]}...")
            
    except Exception as e:
        print(f"   ❌ Error analyzing transaction: {e}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

# ===========================================================================
# SUMMARY OF ALL USERS COHERENCY (Direct tool call example)
# ===========================================================================
print("\n📍 Location Coherency Summary (All Users):")
coherency_result = get_all_users_coherency.invoke({"window_hours": 6, "max_plausible_kmh": 900.0})
print(f"   Average coherence score: {coherency_result.get('average_score', 'N/A')}")
