from typing import TypedDict, Any, Optional, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_ollama import ChatOllama


class EmailState(TypedDict):
    email: dict[str, Any]   # Email information: sender, subject, body
    category: Optional[str] # Category of the email
    is_spam: bool
    spam_reason: Optional[str]
    draft_response: Optional[str]
    messages: Annotated[Sequence[BaseMessage], add_messages]


llm = ChatOllama(model="qwen2.5:3b", temperature=0, client_kwargs={"timeout": 60.0})


def read_email(state: EmailState) -> EmailState:
    """Node to read and process an email."""
    email = state["email"]

    # Some preprocessing steps could go here
    print(f"-> Processing email from {email['sender']} with subject {email['subject']}")

    return {} # No changes in state


def classify_email(state: EmailState) -> EmailState:
    """Node to classify the email based on its content."""
    email = state["email"]

    prompt = f"""
    As mail assistant, analyze this email and determine if it is spam or legitimate.

    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}

    First, determine if this email is spam. If it is spam, explain why.
    If it is legitimate, categorize it (inquiry, complaint, thank you, etc.).
    """

    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)

    # Simple parsing (should be more robust)
    response_text = response.content.lower()
    is_spam = "spam" in response_text and "not spam" not in response_text

    # Extract a reason if it's spam (should be more robust)
    spam_reason = None
    if is_spam and "reason:" in response_text:
        spam_reason = response_text.split("reason:")[1].strip()

    # Extract email category (should be more robust)
    email_category = None
    if not is_spam:
        for category in ["inquiry", "complaint", "thank you", "offer", "request", "information"]:
            if category in response_text:
                email_category = category
                break

    return {
        "is_spam": is_spam,
        "spam_reason": spam_reason,
        "category": email_category,
        "messages": messages + [AIMessage(content=response.content)],
    }


def process_spam(state: EmailState) -> EmailState:
    """Node to process spam emails."""
    print(f"The email has marked as spam. Reason: {state['spam_reason']}")
    print("The email has been moved to the spam folder.")
    return {}


def draft_response(state: EmailState) -> EmailState:
    """Node to draft a response to the email."""
    email = state["email"]
    category = state["category"] or "unknown"

    prompt = f"""
    As email assistant, draft a polite preliminary response to this email.

    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}

    This email has been categorized as: {category}

    Draft a brief, professional response that Mr. Hugg can review and personalize before sending.
    """

    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)

    return {
        "draft_response": response.content,
        "messages": messages + [AIMessage(content=response.content)],
    }


def notification(state: EmailState) -> EmailState:
    """Node to send a notification to the user."""
    email = state["email"]

    print("\n" + "="*50)
    print(f"Sir, you've received an email from {email['sender']}.")
    print(f"Subject: {email['subject']}")
    print(f"Category: {state['category']}")
    print("\nI've prepared a draft response for your review:")
    print("-"*50)
    print(state["draft_response"])
    print("="*50 + "\n")

    return {}


def route_email(state: EmailState) -> str:
    """Route the email based on its classification."""
    if state["is_spam"]:
        return "spam"
    else:
        return "legitimate"


graph = StateGraph(EmailState)

graph.add_node("read_email", read_email)
graph.add_node("classify_email", classify_email)
graph.add_node("process_spam", process_spam)
graph.add_node("draft_response", draft_response)
graph.add_node("notification", notification)

graph.add_edge(START, "read_email")
graph.add_edge("read_email", "classify_email")
graph.add_conditional_edges(
    "classify_email",
    route_email,
    {
        "spam": "process_spam",
        "legitimate": "draft_response",
    }
)
graph.add_edge("draft_response", "notification")
graph.add_edge("process_spam", END)
graph.add_edge("notification", END)

app = graph.compile()

# email = {
#     "sender": "john.smith@example.com",
#     "subject": "Question about your services",
#     "body": "Dear Mr. Hugg, I was referred to you by a colleague and I'm interested in learning more about your consulting services. Could we schedule a call next week? Best regards, John Smith"
# }

email = {
    "sender": "winner@lottery-intl.com",
    "subject": "YOU HAVE WON $5,000,000!!!",
    "body": "CONGRATULATIONS! You have been selected as the winner of our international lottery! To claim your $5,000,000 prize, please send us your bank details and a processing fee of $100."
}

result = app.invoke({
    "email": email,
    "is_spam": None,
    "spam_reason": None,
    "email_category": None,
    "email_draft": None,
    "messages": []
})