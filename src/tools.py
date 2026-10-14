"""
Mock backend "tools" that the agent can call.

These simulate real e-commerce backend APIs so the assistant can demonstrate
agentic tool-calling without needing a real backend. Each function:
  - takes simple JSON-serialisable arguments
  - returns a JSON-serialisable dict
  - never raises for "not found" cases -> returns a structured error the
    agent can turn into a helpful reply instead of crashing.
"""
import json
import uuid
from datetime import datetime, timedelta

from . import config


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_orders():
    return _load_json(config.ORDERS_FILE)


def _load_products():
    return _load_json(config.PRODUCTS_FILE)


def _load_returns_store():
    if config.RETURNS_STORE_FILE.exists():
        return _load_json(config.RETURNS_STORE_FILE)
    return []


def _save_returns_store(returns):
    _save_json(config.RETURNS_STORE_FILE, returns)


# ---------------------------------------------------------------------------
# Tool 1: get_order_status
# ---------------------------------------------------------------------------
def get_order_status(order_id: str) -> dict:
    """Look up the current status of an order by its order ID."""
    order_id = (order_id or "").strip().upper()
    orders = _load_orders()
    for order in orders:
        if order["order_id"].upper() == order_id:
            return {
                "found": True,
                "order_id": order["order_id"],
                "status": order["status"],
                "items": order["items"],
                "order_date": order["order_date"],
                "expected_delivery": order.get("expected_delivery"),
                "tracking_id": order.get("tracking_id"),
            }
    return {
        "found": False,
        "order_id": order_id,
        "message": f"No order found with ID '{order_id}'. Please double check the order ID "
                   f"(it usually looks like ORD1001) and try again.",
    }


# ---------------------------------------------------------------------------
# Tool 2: create_return_request
# ---------------------------------------------------------------------------
def create_return_request(order_id: str, reason: str, resolution: str = "refund") -> dict:
    """
    File a return/replacement request for an order.

    resolution: "refund" or "replacement"
    """
    order_id = (order_id or "").strip().upper()
    orders = _load_orders()
    order = next((o for o in orders if o["order_id"].upper() == order_id), None)

    if order is None:
        return {
            "success": False,
            "message": f"No order found with ID '{order_id}'. Cannot create a return.",
        }

    if order["status"] != "Delivered":
        return {
            "success": False,
            "message": (
                f"Order {order_id} is currently '{order['status']}'. "
                f"Returns can only be requested for orders that have been Delivered. "
                f"If it hasn't shipped yet, it may still be eligible for cancellation instead."
            ),
        }

    delivered_date = order.get("delivered_date")
    if delivered_date:
        days_since = (datetime.now() - datetime.strptime(delivered_date, "%Y-%m-%d")).days
        if days_since > 7:
            return {
                "success": False,
                "message": (
                    f"Order {order_id} was delivered {days_since} days ago, which is outside "
                    f"the 7-day return window. This request should be escalated to a human agent "
                    f"for a case-by-case review."
                ),
            }

    returns = _load_returns_store()
    return_id = f"RET{uuid.uuid4().hex[:8].upper()}"
    record = {
        "return_id": return_id,
        "order_id": order_id,
        "reason": reason,
        "resolution": resolution,
        "status": "Pickup Scheduled",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    returns.append(record)
    _save_returns_store(returns)

    return {
        "success": True,
        "return_id": return_id,
        "order_id": order_id,
        "resolution": resolution,
        "status": record["status"],
        "message": (
            f"Return request {return_id} created for order {order_id}. "
            f"Our courier partner will schedule a pickup. "
            f"{'Refund' if resolution == 'refund' else 'Replacement'} will be processed "
            f"after quality check (typically 3-7 business days for refunds)."
        ),
    }


# ---------------------------------------------------------------------------
# Tool 3: get_refund_policy
# ---------------------------------------------------------------------------
def get_refund_policy() -> dict:
    """Return a structured summary of the refund/return policy (not RAG - a fixed lookup)."""
    return {
        "return_window_days": 7,
        "condition_required": "unused, original condition, with accessories and packaging",
        "refund_timeline_business_days": "3-7 after quality check",
        "non_returnable_examples": [
            "hygiene/personal care items",
            "software licenses and digital subscriptions",
            "items explicitly marked non-returnable",
        ],
    }


# ---------------------------------------------------------------------------
# Tool 4: get_product_info
# ---------------------------------------------------------------------------
def get_product_info(product_id: str = None, product_name: str = None) -> dict:
    """Look up a product by ID or (partial, case-insensitive) name."""
    products = _load_products()

    if product_id:
        pid = product_id.strip().upper()
        for p in products:
            if p["id"].upper() == pid:
                return {"found": True, "product": p}

    if product_name:
        name = product_name.strip().lower()
        matches = [p for p in products if name in p["name"].lower()]
        if matches:
            return {"found": True, "product": matches[0], "other_matches": matches[1:]}

    return {"found": False, "message": "No matching product found in the catalog."}


# ---------------------------------------------------------------------------
# Tool schema definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get the current status, items, and tracking info for a customer's order using their order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID, e.g. ORD1001",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_return_request",
            "description": "File a return or replacement request for a delivered order. Only works if the order status is 'Delivered' and within the 7-day return window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The order ID, e.g. ORD1003"},
                    "reason": {"type": "string", "description": "Customer's reason for the return"},
                    "resolution": {
                        "type": "string",
                        "enum": ["refund", "replacement"],
                        "description": "Whether the customer wants a refund or a replacement",
                    },
                },
                "required": ["order_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_refund_policy",
            "description": "Get a structured summary of the store's refund and return policy (window, timelines, non-returnable items).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_info",
            "description": "Look up product details (price, specs, stock) by product ID or product name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string", "description": "Product ID, e.g. P1001"},
                    "product_name": {"type": "string", "description": "Full or partial product name"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the FAQ and Policies knowledge base (RAG) for information about shipping, returns, cancellations, payments, warranty, or general how-to questions that are NOT about a specific order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The customer's question, or a focused search query derived from it"}
                },
                "required": ["query"],
            },
        },
    },
]

# Dispatch table mapping tool name -> python function.
# search_knowledge_base is wired up separately in agent.py because it needs the retriever instance.
TOOL_FUNCTIONS = {
    "get_order_status": get_order_status,
    "create_return_request": create_return_request,
    "get_refund_policy": get_refund_policy,
    "get_product_info": get_product_info,
}
