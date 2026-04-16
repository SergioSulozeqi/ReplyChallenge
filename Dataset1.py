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
