import os
import json
import re
import ulid
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
#from langchain_core.messages import HumanMessage
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler


import numpy as np
from datetime import datetime
from collections import Counter
from langchain_core.tools import tool
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


# TOOL UserProfileAgent 
# ── TOOL 1: get_user_profile ───────────────────────────────────────────────

@tool
def get_user_profile() -> str:
    """Restituisce il profilo demografico e le abitudini dell'utente:
    nome, lavoro, stipendio, città di residenza, descrizione delle abitudini.
    Utile per contestualizzare le anomalie."""
    profile = _context["user_profile"]
    if not profile:
        return "Profilo utente non disponibile."
    return (
        f"Nome: {profile.get('first_name')} {profile.get('last_name')}\n"
        f"Lavoro: {profile.get('job')}\n"
        f"Stipendio annuo: €{profile.get('salary')}\n"
        f"Città: {profile.get('residence', {}).get('city', 'N/A')}\n"
        f"Descrizione: {profile.get('description', 'N/A')}"
    )

# ── TOOL 2: get_user_history ───────────────────────────────────────────────

@tool
def get_user_history(n: int = 5) -> str:
    """Restituisce le ultime N transazioni dell'utente.
    Utile per capire il comportamento abituale nel tempo."""
    history = _context["history"]
    window  = history[-n:] if len(history) >= n else history
    if not window:
        return "Nessuna transazione precedente disponibile."
    lines = []
    for tx in window:
        lines.append(
            f"[{str(tx['timestamp'])[:10]}] "
            f"tipo={tx['transaction_type']} "
            f"importo={tx['amount']} "
            f"iban_dest={tx.get('recipient_iban', 'N/A')} "
            f"desc={tx.get('description', 'N/A')} "
            f"balance={tx.get('balance_after', 'N/A')}"
        )
    return "\n".join(lines)

# ── TOOL 3: compute_behavioral_stats ──────────────────────────────────────

@tool
def compute_behavioral_stats() -> str:
    """Calcola statistiche comportamentali dell'utente: media/std importi,
    tipi di transazione usati, distribuzione oraria, recipient IBAN frequenti."""
    history = _context["history"]
    if not history:
        return "Storia insufficiente per calcolare statistiche."
    amounts     = [float(tx["amount"]) for tx in history]
    types       = [tx["transaction_type"] for tx in history]
    hours       = [datetime.fromisoformat(str(tx["timestamp"])).hour for tx in history]
    ibans       = [tx.get("recipient_iban", "") for tx in history if tx.get("recipient_iban")]
    stats = {
        "importo_medio":              round(float(np.mean(amounts)), 2),
        "importo_std":                round(float(np.std(amounts)), 2),
        "importo_min":                round(float(np.min(amounts)), 2),
        "importo_max":                round(float(np.max(amounts)), 2),
        "tipi_tx_usati":              dict(Counter(types)),
        "ora_media_tx":               round(float(np.mean(hours)), 1),
        "ora_std_tx":                 round(float(np.std(hours)), 1),
        "recipient_ibans_frequenti":  dict(Counter(ibans)),
        "totale_tx_storia":           len(history),
    }
    return json.dumps(stats, ensure_ascii=False, indent=2)

# ── TOOL 4: check_recipient_known ─────────────────────────────────────────

@tool
def check_recipient_known() -> str:
    """Verifica se il destinatario IBAN della transazione corrente è già
    stato usato in passato. Restituisce se è noto e quante volte."""
    history      = _context["history"]
    current_iban = _context["transaction"].get("recipient_iban", "")
    if not current_iban:
        return "Nessun IBAN destinatario (transazione non è un bonifico)."
    occurrences = sum(1 for tx in history if tx.get("recipient_iban", "") == current_iban)
    if occurrences == 0:
        return f"IBAN {current_iban} MAI visto prima. Destinatario nuovo."
    return f"IBAN {current_iban} già visto {occurrences} volta/e. Destinatario noto."

#agent creation
sms_agent = create_agent(
    model=model,
    system_prompt="When user asks for sms evaluation, use the provided tool to parse the sms.json file." \
    "Give to each text a score form 0 to 1 based on the presence of scam indicators such as urgency in the message." \
    "Return the list of sender, recipient, date, and score for each sms.",
    tools=[get_all_parsed_sms]
)

profile_agent = create_agent(
    model=model,
    system_prompt=r"""Sei un agente specializzato nel rilevamento di frodi bancarie.

Hai questi tool disponibili:
- get_user_profile: profilo demografico e abitudini dell'utente
- get_user_history: ultime N transazioni dell'utente
- compute_behavioral_stats: statistiche su importi (media, std), orari, tipi di tx
- check_recipient_known: verifica se il destinatario IBAN è noto

Usa i tool per raccogliere dati. Se l'importo della transazione è molto diverso dalla media, calcola lo Z-score.
Formula suggerita: $Z = \frac{(x - \mu)}{\sigma}$ (dove x è l'importo attuale, \mu la media e \sigma la deviazione standard).

Produci ESCLUSIVAMENTE questo JSON:
{
  "score": <float 0.0-1.0>,
  "reasoning": "<spiegazione in italiano>",
  "metrics": {
    "amount_zscore": <float>,
    "is_new_recipient": <bool>,
    "deviation_from_avg_amount": <float>,
    "is_new_tx_type": <bool>
  }
}

Linee guida score:
- 0.0-0.3: comportamento normale
- 0.3-0.6: alcune deviazioni
- 0.6-1.0: anomalie gravi""",
    tools=[get_user_profile, get_user_history, compute_behavioral_stats, check_recipient_known]
)

# -----------------------LocationAgent tools----------------------------
# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_data():
    locs = pd.DataFrame(json.loads(Path("locations.json").read_text()))
    locs["timestamp"] = pd.to_datetime(locs["timestamp"])
    
    txns = pd.read_csv("transactions.csv")
    txns["timestamp"] = pd.to_datetime(txns["timestamp"])
    
    users = pd.DataFrame(json.loads(Path("users.json").read_text()))
    
    # Map IBAN to biotag
    iban_to_biotag = {}
    for _, row in txns.iterrows():
        if pd.notna(row.get("sender_iban")) and pd.notna(row.get("sender_id")):
            iban_to_biotag[row["sender_iban"]] = row["sender_id"]
        if pd.notna(row.get("recipient_iban")) and pd.notna(row.get("recipient_id")):
            iban_to_biotag[row["recipient_iban"]] = row["recipient_id"]
    
    users["biotag"] = users["iban"].map(iban_to_biotag)
    return locs, txns, users


_locs, _txns, _users = _load_data()


# ---------------------------------------------------------------------------
# Core distance function
# ---------------------------------------------------------------------------
def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Tool input schemas
# ---------------------------------------------------------------------------
class UserCoherencyInput(BaseModel):
    biotag: str = Field(description="User biotag (e.g., 'RGNR-LNAA-7FF-AUD-0')")
    window_hours: int = Field(6, description="Hours around transaction to search for GPS ping")
    max_plausible_kmh: float = Field(900.0, description="Maximum plausible travel speed in km/h")


class AllUsersCoherencyInput(BaseModel):
    window_hours: int = Field(6, description="Hours around transaction to search for GPS ping")
    max_plausible_kmh: float = Field(900.0, description="Maximum plausible travel speed in km/h")


# ---------------------------------------------------------------------------
# LangChain Tools
# ---------------------------------------------------------------------------
@tool(args_schema=UserCoherencyInput)
def get_user_coherency_score(
    biotag: str,
    window_hours: int = 6,
    max_plausible_kmh: float = 900.0
) -> dict:
    """
    Calculate coherency score (0-1) for a single user based on GPS-transaction alignment.
    Score 1 = perfectly coherent, 0 = completely incoherent.
    """
    user = _users[_users["biotag"] == biotag]
    if user.empty:
        return {"error": f"User {biotag} not found"}
    
    home_lat = float(user.iloc[0]["residence"]["lat"])
    home_lng = float(user.iloc[0]["residence"]["lng"])
    
    # Get user's transactions
    user_txns = _txns[(_txns["sender_id"] == biotag) | (_txns["recipient_id"] == biotag)]
    if user_txns.empty:
        return {"biotag": biotag, "score": 1.0, "reason": "No transactions to verify"}
    
    user_locs = _locs[_locs["biotag"] == biotag]
    violations = 0
    total_weight = 0
    
    for _, txn in user_txns.iterrows():
        # Find nearest GPS ping
        txn_time = txn["timestamp"]
        nearby = user_locs[abs(user_locs["timestamp"] - txn_time) <= timedelta(hours=window_hours)]
        
        weight = 1.0
        if txn["transaction_type"] == "transfer":
            weight = 0.3  # Transfers less location-dependent
        elif txn["transaction_type"] in ["e-commerce", "direct debit"]:
            weight = 0.1  # Online transactions location-independent
        
        total_weight += weight
        
        if nearby.empty:
            violations += weight * 0.5  # Missing GPS = partial violation
            continue
        
        # Check distances
        for _, gps in nearby.iterrows():
            dist = _haversine_km(gps["lat"], gps["lng"], home_lat, home_lng)
            time_diff_h = abs((gps["timestamp"] - txn_time).total_seconds()) / 3600
            
            # Velocity check
            if time_diff_h > 0:
                speed = dist / time_diff_h
                if speed > max_plausible_kmh and dist > 10:
                    violations += weight
                    break
            # Distance check for in-person transactions
            elif txn["transaction_type"] == "in-person payment" and dist > 100:
                violations += weight * 0.8
                break
    
    score = max(0.0, 1.0 - (violations / total_weight if total_weight > 0 else 0))
    return {
        "biotag": biotag,
        "name": f"{user.iloc[0]['first_name']} {user.iloc[0]['last_name']}",
        "coherency_score": round(score, 3),
        "transactions_checked": len(user_txns),
        "violations_weight": round(violations, 2),
        "interpretation": "High coherence" if score > 0.8 else "Medium coherence" if score > 0.5 else "Low coherence"
    }


@tool(args_schema=AllUsersCoherencyInput)
def get_all_users_coherency(
    window_hours: int = 6,
    max_plausible_kmh: float = 900.0
) -> dict:
    """
    Calculate coherency scores (0-1) for all users in the system.
    Returns ranked list with scores and summary statistics.
    """
    results = []
    for biotag in _users["biotag"].dropna().unique():
        result = get_user_coherency_score.invoke({
            "biotag": biotag,
            "window_hours": window_hours,
            "max_plausible_kmh": max_plausible_kmh
        })
        if "error" not in result:
            results.append(result)
    
    results.sort(key=lambda x: x["coherency_score"])
    
    return {
        "total_users": len(results),
        "average_score": round(sum(r["coherency_score"] for r in results) / len(results), 3),
        "lowest_coherency": results[0] if results else None,
        "highest_coherency": results[-1] if results else None,
        "all_scores": results,
        "risk_ranking": [
            {
                "rank": i + 1,
                "biotag": r["biotag"],
                "name": r["name"],
                "score": r["coherency_score"],
                "risk_level": "HIGH" if r["coherency_score"] < 0.5 else "MEDIUM" if r["coherency_score"] < 0.8 else "LOW"
            }
            for i, r in enumerate(results)
        ]
    }


@tool
def get_user_transaction_details(biotag: str) -> dict:
    """
    Get detailed transaction history with location context for a specific user.
    """
    user = _users[_users["biotag"] == biotag]
    if user.empty:
        return {"error": f"User {biotag} not found"}
    
    home = user.iloc[0]["residence"]
    user_txns = _txns[(_txns["sender_id"] == biotag) | (_txns["recipient_id"] == biotag)]
    user_locs = _locs[_locs["biotag"] == biotag]
    
    details = []
    for _, txn in user_txns.iterrows():
        nearby = user_locs[abs(user_locs["timestamp"] - txn["timestamp"]) <= timedelta(hours=6)]
        
        detail = {
            "transaction_id": txn["transaction_id"],
            "timestamp": str(txn["timestamp"]),
            "type": txn["transaction_type"],
            "amount": txn["amount"],
            "location_stated": txn.get("location", "N/A"),
        }
        
        if not nearby.empty:
            closest = nearby.iloc[(nearby["timestamp"] - txn["timestamp"]).abs().argmin()]
            detail["gps_location"] = {
                "city": closest["city"],
                "distance_from_home_km": round(
                    _haversine_km(closest["lat"], closest["lng"], float(home["lat"]), float(home["lng"])), 1
                ),
                "time_diff_hours": round(abs((closest["timestamp"] - txn["timestamp"]).total_seconds()) / 3600, 2)
            }
        else:
            detail["gps_location"] = None
            
        details.append(detail)
    
    return {
        "biotag": biotag,
        "name": f"{user.iloc[0]['first_name']} {user.iloc[0]['last_name']}",
        "home": home,
        "transaction_count": len(details),
        "transactions": details
    }

# -----------------------LocationAgent tools----------------------------




# Initialize Langfuse client
langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")
)

def generate_session_id():
    """Generate a unique session ID using TEAM_NAME and ULID."""
    # session_id must not contain blank spaces; TEAM_NAME may include spaces—replace with "-".
    team = os.getenv("TEAM_NAME", "tutorial").replace(" ", "-")
    return f"{team}-{ulid.new().str}"

def invoke_langchain(model, prompt, langfuse_handler, session_id):
    """Invoke LangChain with the given prompt and Langfuse handler."""
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
def run_llm_call(session_id, model, prompt):
    """Run a single LangChain invocation and track it in Langfuse."""
    # Pass session_id via LangChain metadata for session grouping
    # Create Langfuse callback handler for automatic generation tracking
    # The handler will attach to the current trace created by @observe()
    langfuse_handler = CallbackHandler()

    # Invoke LangChain with Langfuse handler to track tokens and costs
    response = invoke_langchain(model, prompt, langfuse_handler, session_id)

    return response

print("✓ Langfuse initialized successfully")
print(f"✓ Public key: {os.getenv('LANGFUSE_PUBLIC_KEY', 'Not set')[:20]}...")
print("✓ Helper functions ready: generate_session_id(), invoke_langchain(), run_llm_call()")
