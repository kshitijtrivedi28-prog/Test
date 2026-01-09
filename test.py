# ============================================================
# LangGraph + Class-Based Agents (FULL FINAL VERSION)
# ============================================================
# pip install langgraph langchain google-generativeai
# ============================================================

from email import message
import random

import re
from typing import TypedDict, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dataclasses import field
import json
import os

# ============================================================
# 1. Shared State (Message ID restored)
# ============================================================

class TicketState(TypedDict):
    id: str
    channel: str
    raw: Dict[str, Any]
    text: Optional[str]

    # enrichment: Dict[str, Any]
    enrichment: Dict[str, Any] = field(default_factory=dict)

    llm_raw: Optional[str]
    intent: Optional[str]
    intent_summary: Optional[str]
    intent_confidence: Optional[float]
    sentiment: Optional[str]
    sentiment_score: Optional[float]
    intent_source: Optional[str]

    route: Optional[str]
    outcome: Optional[Dict[str, Any]]


# ============================================================
# 2. LLM Setup
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key="AIzaSyCD_D8yn-vORbduPKoPJtHAgRXuzSMHuB8"  # REQUIRED
)

# ============================================================
# 3. Base Agent
# ============================================================

class BaseAgent:
    def run(self, state: TicketState) -> TicketState:
        raise NotImplementedError


# ============================================================
# 4. Intake Agents (MATCH OLD CODE STRUCTURE)
# ============================================================

class VoiceIntakeAgent(BaseAgent):
    def run(self, state: TicketState) -> TicketState:
        stt = state["raw"].get("audio_transcript") or "simulated voice: pump stopped"
        return {
            **state,
            "text": stt,
            "enrichment": {
                **state.get("enrichment", {}),
                "confidence": 0.90
            }
        }


class EmailIntakeAgent(BaseAgent):
    def run(self, state: TicketState) -> TicketState:
        raw = state["raw"]
        text = f"{raw.get('subject','')}\n\n{raw.get('body','')}".strip()
        return {**state, "text": text}


class ChatIntakeAgent(BaseAgent):
    def run(self, state: TicketState) -> TicketState:
        return {**state, "text": state["raw"].get("text", "")}


class PortalIntakeAgent(BaseAgent):
    def run(self, state: TicketState) -> TicketState:
        return {**state, "text": state["raw"].get("issue_description", "")}


# ============================================================
# 5. Intent Agent (STRICT JSON + fallback)
# ============================================================

class IntentAgent(BaseAgent):
    SYSTEM_PROMPT = """
You are an intent-and-sentiment extraction assistant. For any user message, output ONLY a single JSON object with these fields:
- intent: a 1-6 word concise natural-language description of the user's goal (use snake_case).
- summary: one short sentence (<=18 words) explaining the intent.
- intent_confidence: number between 0 and 1 (model's confidence in intent).
- sentiment: one of "positive", "neutral", or "negative".
- sentiment_score: number between 0 and 1 (model's confidence in sentiment).
If the message is empty or nonsensical, set intent="unknown", intent_confidence=0, sentiment="neutral", sentiment_score=0.


FEW_SHOT_EXAMPLES_INTENT = [
    {
        "user": "Where is my order? It's late and I'm upset.",
        "assistant": {
            "intent":"check_order_status",
            "summary":"User asks for current status/ETA of an order.",
            "intent_confidence":0.92,
            "sentiment":"negative",
            "sentiment_score":0.95
        }
    },
    {
        "user": "Thanks! That fixed my issue.",
        "assistant": {
            "intent":"confirm_resolution",
            "summary":"User confirms the problem is resolved and expresses thanks.",
            "intent_confidence":0.9,
            "sentiment":"positive",
            "sentiment_score":0.9
        }
    }
]

"""

    def __init__(self, llm):
        self.llm = llm

    def run(self, state: TicketState) -> TicketState:
        prompt = f"{self.SYSTEM_PROMPT}\n\nMessage:\n{state['text']}"

        try:
            
            raw = self.llm.invoke(prompt).content
        
            raw = raw.strip()

    # Extract first {...} block if model adds text
            match = re.search(r"\{(?:[^{}]|\{[^{}]*\})*\}", raw, re.S)
            if not match:
                raise ValueError("No JSON object found in LLM output")

            data = json.loads(match.group(0))
            # data.enrichment["customer_id"] = state["raw"].get("customer_id", "CUST-0001")
            warranty = random.choice(["in_warranty", "out_of_warranty"]) 
                     
            # print("data", data)
            
            return {
                **state,
                "llm_raw": raw,
                "intent": data["intent"],
                "enrichment": {"customer_id": state["raw"].get("customer_id", "CUST-0001"), "warranty_status": warranty},
                # "enrichment": {"warranty_status": warranty},
                "intent_summary": data["summary"],
                "intent_confidence": data["intent_confidence"],
                "sentiment": data["sentiment"],
                "sentiment_score": data["sentiment_score"],
                "intent_source": "llm",
            }

        except Exception as e:
            text = (state["text"] or "").lower()
            print("⚠️ LLM intent extraction failed:", e)
            if any(k in text for k in ("defect", "fault", "broken", "stopped")):
                intent = "quality_complaint_dummy"
            elif any(k in text for k in ("install", "setup")):
                intent = "installation_dummy"
            elif "calibration" in text:
                intent = "calibration_dummy"
            elif any(k in text for k in ("maintenance", "pm")):
                intent = "maintenance_dummy"
            else:
                intent = "general_query_dummy"

            return {
                **state,
                "intent": intent,
                "intent_summary": "Rule-based fallback",
                "intent_confidence": 0.0,
                "sentiment": "neutral",
                "sentiment_score": 0.0, 
                "intent_source": "fallback",
            }


# ============================================================
# 6. Routing Agent
# ============================================================

class RoutingAgent(BaseAgent):
    def run(self, state: TicketState) -> TicketState:
        intent = state["intent"]

        if intent == "quality_complaint":
            route = "trackwise"
        elif intent in (
            "installation",
            "maintenance",
            "calibration",
            "preventive_maintenance",
        ):
            route = "servicemax"
        else:
            route = "salesforce"

        return {**state, "route": route}


# ============================================================
# 7. Backend Agents
# ============================================================

class SalesforceAgent(BaseAgent):
    def run(self, state: TicketState) -> TicketState:
        return {**state, "outcome": {"salesforce_case": "SF-10021"}}


class ServiceMaxAgent(BaseAgent):
    def run(self, state: TicketState) -> TicketState:
        return {**state, "outcome": {"work_order": "SM-45012"}}


class TrackWiseAgent(BaseAgent):
    def run(self, state: TicketState) -> TicketState:
        return {**state, "outcome": {"capa": "CAPA-77881"}}


# ============================================================
# 8. Response Agent (TRACEABLE BY ID)
# ============================================================

class ResponseAgent(BaseAgent):
    def run(self, state: TicketState) -> TicketState:
        
        print("\n========== FINAL RESULT ==========")
        print(f"Message ID     : {state['id']}")
        print(f"Channel        : {state['channel']}")
        print(f"Text           : {state['text']}")
        print(f"Intent         : {state['intent']}")
        print(f"Intent Source  : {state['intent_source']}")
        print(f"Confidence     : {state.get('intent_confidence')}")
        print(f"Sentiment Score : {state.get('sentiment_score')}")
        print(f"Sentiment      : {state.get('sentiment')}")
        print(f"Customer ID   : {state['enrichment'].get('customer_id')}")
        print(f"Warranty Status: {state['enrichment'].get('warranty_status')}")
        print(f"Route          : {state['route']}")
        print(f"Outcome        : {state['outcome']}")
        print("=================================\n")
        return state


# ============================================================
# 9. LangGraph Wiring
# ============================================================

voice_intake = VoiceIntakeAgent()
email_intake = EmailIntakeAgent()
chat_intake = ChatIntakeAgent()
portal_intake = PortalIntakeAgent()

intent_agent = IntentAgent(llm)
routing_agent = RoutingAgent()

sf_agent = SalesforceAgent()
sm_agent = ServiceMaxAgent()
tw_agent = TrackWiseAgent()

response_agent = ResponseAgent()

graph = StateGraph(TicketState)

graph.add_node("intake_router", lambda s: s)
graph.add_node("intake_voice", voice_intake.run)
graph.add_node("intake_email", email_intake.run)
graph.add_node("intake_chat", chat_intake.run)
graph.add_node("intake_portal", portal_intake.run)

graph.add_node("intent", intent_agent.run)
graph.add_node("route", routing_agent.run)

graph.add_node("salesforce", sf_agent.run)
graph.add_node("servicemax", sm_agent.run)
graph.add_node("trackwise", tw_agent.run)

graph.add_node("response", response_agent.run)

graph.set_entry_point("intake_router")

def select_intake(state: TicketState):
    return f"intake_{state['channel']}"

graph.add_conditional_edges(
    "intake_router",
    select_intake,
    {
        "intake_voice": "intake_voice",
        "intake_email": "intake_email",
        "intake_chat": "intake_chat",
        "intake_portal": "intake_portal",
    },
)

graph.add_edge("intake_voice", "intent")
graph.add_edge("intake_email", "intent")
graph.add_edge("intake_chat", "intent")
graph.add_edge("intake_portal", "intent")

graph.add_edge("intent", "route")

def select_route(state: TicketState):
    return state["route"]

graph.add_conditional_edges(
    "route",
    select_route,
    {
        "salesforce": "salesforce",
        "servicemax": "servicemax",
        "trackwise": "trackwise",
    },
)

graph.add_edge("salesforce", "response")
graph.add_edge("servicemax", "response")
graph.add_edge("trackwise", "response")
graph.add_edge("response", END)

app = graph.compile()


# ============================================================
# 10. Input Simulation (OLD STYLE IDs RESTORED)
# ============================================================

if __name__ == "__main__":

    inputs = [
        {
            "id": "m-001",
            "channel": "email",
            "raw": {
                "subject": "Pump Fault - URGENT",
                "body": "Pump stopped. SN: 7890.",
                "attachments": []
            },
        },
        {
            "id": "m-002",
            "channel": "voice",
            "raw": {
                "audio_transcript": "Installation required SN: 4567"
            },
        },
        {
            "id": "m-003",
            "channel": "chat",
            "raw": {
                "text": "Calibration required please assist"
            },
        },
        {
            "id": "m-004",
            "channel": "portal",
            "raw": {
                "issue_description": "Motor defect",
                "product": "Pump Z"
            },
        },
    ]

    for i in inputs:
        app.invoke(
            {
                "id": i["id"],
                "channel": i["channel"],
                "raw": i["raw"],
                "text": None,
                "enrichment": {},
                "llm_raw": None,
                "intent": None,
                "intent_summary": None,
                "intent_confidence": None,
                "sentiment": None,
                "intent_source": None,
                "route": None,
                "outcome": None,
            }
        )





        #ascdkaskjcbasbcaskjcbsa