#!/usr/bin/env python3
import os
import sys
import json
import argparse
import re
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv

# Lazy imports for providers to avoid requiring all SDKs when not used

def _try_import_openai():
	try:
		from openai import OpenAI  # type: ignore
		return OpenAI
	except Exception:
		return None


def _try_import_anthropic():
	try:
		import anthropic  # type: ignore
		return anthropic
	except Exception:
		return None


def _try_import_mistral():
	try:
		from mistralai.client import MistralClient  # type: ignore
		from mistralai.models.chat_completion import ChatMessage  # type: ignore
		return MistralClient, ChatMessage
	except Exception:
		return None, None


ALLOWED_CATEGORIES = {"Academic", "HR", "Finance", "IT", "General"}
ALLOWED_URGENCY = {"Critical", "High", "Normal"}


SYSTEM_INSTRUCTIONS = (
	"You are an email triage assistant. Extract a 5-line summary, category, urgency, and action_required. "
	"Output MUST be only a strict JSON object with keys: "
	"summary_lines (array of 5 strings), category (one of Academic, HR, Finance, IT, General), "
	"urgency (one of Critical, High, Normal), action_required (boolean). No additional text."
)


def build_prompt(email_text: str) -> str:
	return (
		"Analyze the following raw email and extract the required fields.\n\n"
		"Email:\n" + email_text.strip() + "\n\n"
		"Return ONLY JSON with: {\n"
		"  \"summary_lines\": [\"...\" x5],\n"
		"  \"category\": \"Academic|HR|Finance|IT|General\",\n"
		"  \"urgency\": \"Critical|High|Normal\",\n"
		"  \"action_required\": true|false\n"
		"}"
	)


def parse_strict_json(text: str) -> Optional[Dict[str, Any]]:
	try:
		return json.loads(text)
	except Exception:
		# Try to extract the first JSON object using a simple regex
		match = re.search(r"\{[\s\S]*\}", text)
		if match:
			try:
				return json.loads(match.group(0))
			except Exception:
				return None
		return None


def heuristic_fallback(email_text: str) -> Dict[str, Any]:
	text = email_text.strip()
	lines = [line.strip() for line in text.splitlines() if line.strip()]
	# Build 5-line summary heuristically
	summary_lines: List[str] = []
	if lines:
		# First line as subject-like
		summary_lines.append(lines[0][:200])
		# Add next distinct informative lines
		for line in lines[1:]:
			if len(summary_lines) >= 5:
				break
			if len(line) > 10:
				summary_lines.append(line[:200])
	while len(summary_lines) < 5:
		summary_lines.append("")

	lower = text.lower()
	category = "General"
	if any(k in lower for k in ["invoice", "payment", "bill", "receipt", "quote", "refund", "balance"]):
		category = "Finance"
	elif any(k in lower for k in ["professor", "assignment", "course", "university", "exam", "grade", "lecture"]):
		category = "Academic"
	elif any(k in lower for k in ["it support", "server", "outage", "password", "ticket", "vpn", "deployment", "bug", "incident"]):
		category = "IT"
	elif any(k in lower for k in ["hr", "benefit", "payroll", "recruit", "hiring", "leave", "policy", "termination"]):
		category = "HR"

	urgency = "Normal"
	if any(k in lower for k in ["urgent", "asap", "immediately", "critical", "production down", "p0", "p1"]):
		urgency = "High"
	if any(k in lower for k in ["security breach", "ransomware", "data loss", "system down", "severe outage", "deadline today"]):
		urgency = "Critical"

	action_required = any(k in lower for k in ["please respond", "action required", "approve", "review", "confirm", "sign", "pay", "submit"])

	return {
		"summary_lines": summary_lines[:5],
		"category": category,
		"urgency": urgency,
		"action_required": action_required,
	}


def call_openai(prompt: str, model: Optional[str]) -> Optional[str]:
	OpenAI = _try_import_openai()
	if OpenAI is None:
		return None
	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		return None
	client = OpenAI(api_key=api_key)
	model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
	try:
		resp = client.chat.completions.create(
			model=model_name,
			messages=[
				{"role": "system", "content": SYSTEM_INSTRUCTIONS},
				{"role": "user", "content": prompt},
			],
			temperature=0.2,
			max_tokens=600,
		)
		return resp.choices[0].message.content or None
	except Exception:
		return None


def call_anthropic(prompt: str, model: Optional[str]) -> Optional[str]:
	anthropic = _try_import_anthropic()
	if anthropic is None:
		return None
	api_key = os.getenv("ANTHROPIC_API_KEY")
	if not api_key:
		return None
	client = anthropic.Anthropic(api_key=api_key)
	model_name = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
	try:
		resp = client.messages.create(
			model=model_name,
			max_tokens=800,
			temperature=0.2,
			messages=[{"role": "user", "content": prompt}],
		)
		# Text is in content list
		for block in resp.content:
			if getattr(block, "type", "") == "text":
				return block.text
		return None
	except Exception:
		return None


def call_mistral(prompt: str, model: Optional[str]) -> Optional[str]:
	MistralClient, ChatMessage = _try_import_mistral()
	if MistralClient is None:
		return None
	api_key = os.getenv("MISTRAL_API_KEY")
	if not api_key:
		return None
	client = MistralClient(api_key=api_key)
	model_name = model or os.getenv("MISTRAL_MODEL", "open-mistral-8x7b")
	try:
		resp = client.chat(
			model=model_name,
			messages=[ChatMessage(role="user", content=prompt)],
			temperature=0.2,
		)
		return resp.choices[0].message.content if resp.choices else None
	except Exception:
		return None


def classify_with_llm(email_text: str, provider: str, model: Optional[str]) -> Dict[str, Any]:
	prompt = build_prompt(email_text)
	raw = None
	if provider == "openai":
		raw = call_openai(prompt, model)
	elif provider == "anthropic":
		raw = call_anthropic(prompt, model)
	elif provider == "mistral":
		raw = call_mistral(prompt, model)
	else:
		raw = None

	if not raw:
		return heuristic_fallback(email_text)

	parsed = parse_strict_json(raw)
	if not parsed:
		return heuristic_fallback(email_text)

	# Validate and coerce
	summary_lines = parsed.get("summary_lines") or []
	if not isinstance(summary_lines, list):
		summary_lines = []
	summary_lines = [str(x) for x in summary_lines][:5]
	while len(summary_lines) < 5:
		summary_lines.append("")

	category = str(parsed.get("category", "General"))
	if category not in ALLOWED_CATEGORIES:
		category = "General"

	urgency = str(parsed.get("urgency", "Normal"))
	if urgency not in ALLOWED_URGENCY:
		urgency = "Normal"

	action_required = bool(parsed.get("action_required", False))

	return {
		"summary_lines": summary_lines,
		"category": category,
		"urgency": urgency,
		"action_required": action_required,
	}


def format_output(result: Dict[str, Any]) -> str:
	# Join the 5-line summary into a single line with separators, as the required output is a single line
	summary = " | ".join((result.get("summary_lines") or [])[:5])
	category = result.get("category", "General")
	urgency = result.get("urgency", "Normal")
	action = "Yes" if result.get("action_required", False) else "No"
	return f"Summary: {summary} Category: {category} Urgency: {urgency} Action: {action}"


def maybe_store_to_google_sheets(result_line: str, structured: Dict[str, Any]) -> None:
	# Optional integration via Bolt's Google Sheets connector can be proxied by a webhook
	if os.getenv("BOLT_SHEETS_ENABLED", "0") != "1":
		return
	url = os.getenv("BOLT_SHEETS_URL", "").strip()
	if not url:
		return
	payload = {
		"line": result_line,
		"summary_lines": structured.get("summary_lines"),
		"category": structured.get("category"),
		"urgency": structured.get("urgency"),
		"action": "Yes" if structured.get("action_required") else "No",
	}
	try:
		import requests  # type: ignore
		headers = {"Content-Type": "application/json"}
		bearer = os.getenv("BOLT_SHEETS_AUTH_BEARER", "").strip()
		if bearer:
			headers["Authorization"] = f"Bearer {bearer}"
		requests.post(url, json=payload, headers=headers, timeout=10)
	except Exception:
		# Silently ignore sheets errors to avoid blocking CLI output
		pass


def read_email_input(file_path: Optional[str]) -> str:
	if file_path:
		with open(file_path, "r", encoding="utf-8") as f:
			return f.read()
	# Else read stdin
	if sys.stdin.isatty():
		raise SystemExit("No input provided. Pass a file path or pipe email text via stdin.")
	return sys.stdin.read()


def main() -> None:
	load_dotenv()
	parser = argparse.ArgumentParser(description="Smart Notification Agent for email processing")
	parser.add_argument("path", nargs="?", help="Path to file containing raw email text. If omitted, reads from stdin.")
	parser.add_argument("--provider", choices=["openai", "anthropic", "mistral"], default=os.getenv("MODEL_PROVIDER", "openai"))
	parser.add_argument("--model", default=None, help="Override model name for the selected provider")
	args = parser.parse_args()

	email_text = read_email_input(args.path)
	result = classify_with_llm(email_text, args.provider, args.model)
	line = format_output(result)
	# Print the EXACT required one-line output
	print(line)
	# Optionally push to Google Sheets via Bolt webhook
	maybe_store_to_google_sheets(line, result)


if __name__ == "__main__":
	main()