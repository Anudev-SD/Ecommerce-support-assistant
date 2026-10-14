"""
Agent orchestration layer.

Implements a standard tool-calling loop against the OpenAI Chat Completions
API:
  1. Send conversation + tool schemas to the model.
  2. If the model requests tool call(s), execute them locally and feed the
     results back.
  3. Repeat until the model returns a final text answer (or we hit the
     MAX_TOOL_HOPS safety limit).

Also implements simple guardrails:
  - A system prompt that constrains tone, scope, and honesty ("say you don't
    know rather than guessing").
  - A confidence/fallback check: if RAG retrieval comes back empty for a
    knowledge question, the model is instructed to say so and offer escalation
    rather than hallucinate an answer.
  - An explicit escalate_to_human "virtual tool" the model can invoke when it
    judges it cannot resolve the issue (e.g. abusive/urgent/out-of-scope).
"""
import json

from openai import OpenAI

from . import config, tools
from .rag import Retriever

SYSTEM_PROMPT = """You are Aria, a customer support assistant for an e-commerce store.

SCOPE: You help with orders, shipping, returns/refunds, cancellations, payments,
warranty, and product questions for THIS store only.

RULES:
1. Ground every factual claim about policies, shipping fees, timelines, or products
   in a tool result (search_knowledge_base, get_order_status, get_refund_policy, or
   get_product_info). Never invent policy numbers, dates, or order details.
2. If a question needs an order ID and the customer hasn't given one, ask for it
   before calling get_order_status.
3. If search_knowledge_base returns no good results, say plainly that you don't have
   that information and offer to escalate to a human agent. Do not guess.
4. Escalate to a human agent (call escalate_to_human) when:
   - the customer explicitly asks for a human/agent/manager
   - the issue is outside your scope (e.g. legal threats, fraud, account hacking,
     safety issues, complaints about a support agent)
   - the customer is repeating the same unresolved issue a third time
   - you cannot find relevant information after searching
5. Be concise, warm, and professional. Avoid corporate filler. Use short paragraphs
   or bullet points for multi-step instructions.
6. Never ask for full card numbers, passwords, or OTPs.
7. If the customer message is abusive, stay calm and professional; do not mirror
   the tone. Offer escalation if the situation is not resolvable via chat.
"""

ESCALATE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "escalate_to_human",
        "description": "Escalate the conversation to a human support agent. Use this when the issue is out of scope, unresolved, sensitive, or the customer explicitly requests a human.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Brief reason for escalation"},
            },
            "required": ["reason"],
        },
    },
}


class SupportAgent:
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or config.OPENAI_API_KEY)
        self.retriever = Retriever(client=self.client)
        self.all_tool_schemas = tools.TOOL_SCHEMAS + [ESCALATE_TOOL_SCHEMA]

    # -- tool execution ---------------------------------------------------
    def _execute_tool(self, name: str, arguments: dict) -> dict:
        if name == "search_knowledge_base":
            results = self.retriever.retrieve(arguments.get("query", ""))
            if not results:
                return {"found": False, "message": "No relevant information found in the knowledge base."}
            return {"found": True, "results": results}

        if name == "escalate_to_human":
            return {
                "escalated": True,
                "reason": arguments.get("reason", "Not specified"),
                "message": f"{config.ESCALATION_PHRASE}. Reason: {arguments.get('reason', 'unspecified')}.",
            }

        fn = tools.TOOL_FUNCTIONS.get(name)
        if fn is None:
            return {"error": f"Unknown tool '{name}'"}
        return fn(**arguments)

    # -- main entrypoint ----------------------------------------------------
    def chat(self, messages: list) -> dict:
        """
        messages: list of {"role": "user"|"assistant", "content": str} representing
                  the conversation so far (no system message - added automatically).

        Returns: {
          "reply": str,
          "tool_trace": [ {"tool": name, "arguments": {...}, "result": {...}}, ... ],
          "escalated": bool,
        }
        """
        api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        tool_trace = []
        escalated = False

        for _hop in range(config.MAX_TOOL_HOPS):
            response = self.client.chat.completions.create(
                model=config.CHAT_MODEL,
                messages=api_messages,
                tools=self.all_tool_schemas,
                tool_choice="auto",
                temperature=0.2,
            )
            choice = response.choices[0]
            msg = choice.message

            if not msg.tool_calls:
                # Final answer.
                return {
                    "reply": msg.content or "",
                    "tool_trace": tool_trace,
                    "escalated": escalated,
                }

            # Model wants to call one or more tools.
            api_messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    arguments = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}

                result = self._execute_tool(name, arguments)
                if name == "escalate_to_human":
                    escalated = True

                tool_trace.append({"tool": name, "arguments": arguments, "result": result})

                api_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    }
                )

        # Safety fallback if we somehow loop too many times.
        return {
            "reply": (
                "I'm having trouble resolving this automatically. "
                f"{config.ESCALATION_PHRASE} who can help further."
            ),
            "tool_trace": tool_trace,
            "escalated": True,
        }
